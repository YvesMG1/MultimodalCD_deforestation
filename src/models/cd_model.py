

### standard imports
import os
import sys
from pathlib import Path


### pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

### sklearn imports
from sklearn.metrics import confusion_matrix

### Set base path ###
base_path = Path(os.getcwd())
while not (base_path / '.git').exists():
    base_path = base_path.parent

### custom imports
sys.path.append(str(base_path / 'src/data/datasets'))
sys.path.append(str(base_path / 'src/evaluation'))
sys.path.append(str(base_path / 'src/models'))
sys.path.append(str(base_path / 'src/visualization'))

from pixel_metrics import ConfuseMatrixMeter
from polygon_metrics import PolygonConfuseMatrixMeter, polygonize
from fully_conv_models import Unet, FCSiamConc, FCSiamDiff, SiamUnet_conc_multi
from changeformer import ChangeFormerV8, ChangeFormerV9, ChangeFormerV10



class CDModel(pl.LightningModule):
    def __init__(self, 
                model: str = "unet",
                in_channels_S2: int = 7,
                in_channels_S1: int = 2,
                num_classes: int = 2,
                deforest_weight: float = 0.5,
                loss: str = "CrossEntropyLoss",
                lr: float = 0.001,
                weight_decay: float = 0.0001,
                patience: int = 10,
                scheduler: str = "StepLR",
                sentinel_type: str = "S2",
                kernel_size: int = 3,
                dropout: float = 0.2,
                iou_threshold: float = 0.05,
                use_custom_val: bool = False,
                use_custom_test: bool = False,
                save_test_results: bool = False,
                return_results: bool = False,
                plot_indices: list = []
            ):
        super(CDModel, self).__init__()
        self.save_hyperparameters()
        self.iou_threshold = iou_threshold
        self.pixel_conf_matrix_meter = ConfuseMatrixMeter(n_class=num_classes)
        self.poly_conf_matrix_meter = PolygonConfuseMatrixMeter(iou_threshold=iou_threshold)

        self.configure_models()
        self.configure_losses()

        self.test_images_A = []
        self.test_images_A2 = []
        self.test_images_B = []
        self.test_images_B2 = []
        self.test_preds = []
        self.test_preds_probs = []
        self.test_targets = []
        self.test_clouds_A = []
        self.test_clouds_B = []
        self.paths = []

    def training_step(self, batch, batch_idx):

        if self.hparams['sentinel_type']  in ["S2", "S1"]:
            imageA, imageB, targets = batch["A"], batch["B"], batch["mask"]
        else:
            imageA, imageB, imageA2, imageB2, targets = batch["A"], batch["B"], batch["A2"], batch["B2"], batch["mask"]

        model: str = self.hparams["model"]
        if model.split("_")[0] == "Unet":
            if self.hparams['sentinel_type'] in ["S2", "S1"]:
                x = torch.cat([imageA, imageB], dim=1)
            else:
                x = torch.cat([imageA, imageB, imageA2, imageB2], dim=1)
            outputs = self(x)
        elif model in ["FCSiamDiff", "FCSiamConc", "ChangeformerV8", "ChangeformerV9", "ChangeformerV10"]:
            if self.hparams['sentinel_type'] == "S2":
                outputs = self.model(x1_S2=imageA, x2_S2=imageB)
            elif self.hparams['sentinel_type'] == "S1":
                outputs = self.model(x1_S1=imageA, x2_S1=imageB)
            else:
                outputs = self.model(imageA, imageB, imageA2, imageB2)
        elif model in ["SiamUnet_conc_multi", "SiamUnet_diff_multi"]:
            outputs = self.model(imageA, imageB, imageA2, imageB2)
        else:
            raise ValueError(f"Model {model} not recognized")

        loss = self.criterion(outputs, targets)
        batch_size = targets.size(0)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss


    def evaluation_step(self, batch, batch_idx):
        
        if self.hparams['sentinel_type']  in ["S2", "S1"]:
            imageA, imageB, targets = batch["A"], batch["B"], batch["mask"]
        else:
            imageA, imageB, imageA2, imageB2, targets = batch["A"], batch["B"], batch["A2"], batch["B2"], batch["mask"]

        model: str = self.hparams["model"]
        if model.split("_")[0] == "Unet":
            if self.hparams['sentinel_type'] in ["S2", "S1"]:
                x = torch.cat([imageA, imageB], dim=1)
            else:
                x = torch.cat([imageA, imageB, imageA2, imageB2], dim=1)
            outputs = self(x)
        elif model in ["FCSiamDiff", "FCSiamConc", "ChangeformerV8", "ChangeformerV9", "ChangeformerV10"]:
            if self.hparams['sentinel_type'] == "S2":
                outputs = self.model(x1_S2=imageA, x2_S2=imageB)
            elif self.hparams['sentinel_type'] == "S1":
                outputs = self.model(x1_S1=imageA, x2_S1=imageB)
            else:
                outputs = self.model(imageA, imageB, imageA2, imageB2)
        elif model in ["SiamUnet_conc_multi", "SiamUnet_diff_multi"]:
            outputs = self.model(imageA, imageB, imageA2, imageB2)
        else:
            raise ValueError(f"Model {model} not recognized")
        
        loss = self.criterion(outputs, targets)

        # Update pixel confusion matrix
        preds = outputs.argmax(dim=1).cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        cm = confusion_matrix(targets_np, preds, labels=list(range(self.pixel_conf_matrix_meter.n_class)))
        self.pixel_conf_matrix_meter.update(cm)

        # Update polygon confusion matrix
        preds_np = outputs.argmax(dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        for image in range(preds_np.shape[0]):
            preds = preds_np[image]
            targets = targets_np[image]
            poly_pred = polygonize(preds)
            poly_actual = polygonize(targets)
            self.poly_conf_matrix_meter.update(poly_pred, poly_actual)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.evaluation_step(batch, batch_idx)
        batch_size = batch["mask"].size(0)  
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_end(self):
        pixel_scores = self.get_pixel_scores()
        poly_scores = self.get_poly_scores()
        self.log('mf1_pix', pixel_scores['mf1'], prog_bar=True, logger=True, on_epoch=True)
        self.log('f11_pix', pixel_scores['F1_1'], prog_bar=True, logger=True, on_epoch=True)
        self.log('mf1_poly', poly_scores['mf1'], prog_bar=True, logger=True, on_epoch=True)
        self.log('iou_poly', poly_scores['iou'], prog_bar=True, logger=True, on_epoch=True)
        self.reset_metrics()

    def test_step(self, batch, batch_idx):
        if self.hparams['use_custom_test']:
            results =  self.test_on_images(batch, batch_idx)
            if results is not None:
                if self.hparams['sentinel_type'] in ["S2", "S1"]:
                    imageA, imageB, preds, pred_probs, targets, cloudsA, cloudsB, path = results
                else:
                    imageA, imageB, imageA2, imageB2, preds, pred_probs, targets, cloudsA, cloudsB, path = results
                
                self.test_images_A.append(imageA)
                self.test_images_A2.append(imageA2) if self.hparams['sentinel_type'] == "both" else None
                self.test_images_B.append(imageB)
                self.test_images_B2.append(imageB2) if self.hparams['sentinel_type'] == "both" else None
                self.test_preds.append(preds)
                self.test_preds_probs.append(pred_probs)
                self.test_targets.append(targets)
                self.test_clouds_A.append(cloudsA)
                self.test_clouds_B.append(cloudsB)
                self.paths.append(path)
        else:
            return self.evaluation_step(batch, batch_idx)

    def test_on_images(self, batch, batch_idx):
        
        if batch_idx in self.hparams['plot_indices']:
            if self.hparams['sentinel_type']  in ["S2", "S1"]:
                imageA, imageB, targets, path = batch["A"], batch["B"], batch["mask"], batch["path"]
            else:
                imageA, imageB, imageA2, imageB2, targets, path = batch["A"], batch["B"], batch["A2"], batch["B2"], batch["mask"], batch["path"]

            model: str = self.hparams["model"]
            if model.split("_")[0] == "Unet":
                if self.hparams['sentinel_type'] in ["S2", "S1"]:
                    x = torch.cat([imageA, imageB], dim=1)
                else:
                    x = torch.cat([imageA, imageB, imageA2, imageB2], dim=1)
                outputs = self(x)
            elif model in ["FCSiamDiff", "FCSiamConc", "ChangeformerV8", "ChangeformerV9", "ChangeformerV10"]:
                if self.hparams['sentinel_type'] == "S2":
                    outputs = self.model(x1_S2=imageA, x2_S2=imageB)
                elif self.hparams['sentinel_type'] == "S1":
                    outputs = self.model(x1_S1=imageA, x2_S1=imageB)
                else:
                    outputs = self.model(imageA, imageB, imageA2, imageB2)
            elif model in ["SiamUnet_conc_multi", "SiamUnet_diff_multi"]:
                outputs = self.model(imageA, imageB, imageA2, imageB2)
            else:
                raise ValueError(f"Model {model} not recognized")

            preds = torch.argmax(outputs, dim=1)
            pred_probs = torch.nn.functional.softmax(outputs, dim=1)
            preds, targets = preds.squeeze(0).cpu().numpy(), targets.squeeze(0).cpu().numpy()
            imageA, imageB = imageA.squeeze(0).cpu().numpy(), imageB.squeeze(0).cpu().numpy()

            cloudsA = batch["cloud_mask_A"].squeeze(0).cpu().numpy() if "cloud_mask_A" in batch else None
            cloudsB = batch["cloud_mask_B"].squeeze(0).cpu().numpy() if "cloud_mask_B" in batch else None

            if self.hparams['sentinel_type'] in ["S2", "S1"]:
                return imageA, imageB, preds, pred_probs, targets, cloudsA, cloudsB, path
            else:
                imageA2, imageB2 = imageA2.squeeze(0).cpu().numpy(), imageB2.squeeze(0).cpu().numpy()
                return imageA, imageB, imageA2, imageB2, preds, pred_probs, targets, cloudsA, cloudsB, path
        return None


    def on_test_epoch_end(self):
        if not self.hparams['use_custom_test']:
            pixel_scores = self.get_pixel_scores()
            poly_scores = self.get_poly_scores()
            self.log('test_mean_f1_pixel', pixel_scores['mf1'], prog_bar=False, logger=True)
            self.log('test_mean_iou_pixel', pixel_scores['miou'], prog_bar=False, logger=True)
            self.log('test_f1_1_pixel', pixel_scores['F1_1'], prog_bar=False, logger=True)
            self.log('test_f1_0_pixel', pixel_scores['F1_0'], prog_bar=False, logger=True)
            self.log('test_accuracy_pixel', pixel_scores['acc'], prog_bar=False, logger=True)
            self.log('test_f1_1_poly', poly_scores['F1_1'], prog_bar=False, logger=True)
            self.log('test_f1_0_poly', poly_scores['F1_0'], prog_bar=False, logger=True)
            self.log('test_mean_f1_poly', poly_scores['mf1'], prog_bar=False, logger=True)
            self.log('test_accuracy_poly', poly_scores['acc'], prog_bar=False, logger=True)
            self.log('test_mean_iou_poly', poly_scores['iou'], prog_bar=False, logger=True)

            return pixel_scores, poly_scores
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        if self.hparams["scheduler"] == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams["patience"], gamma=0.3)
        elif self.hparams["scheduler"] == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=self.hparams["patience"], verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def get_pixel_scores(self):
        return self.pixel_conf_matrix_meter.get_scores()

    def get_poly_scores(self):
        return self.poly_conf_matrix_meter.get_scores()

    def reset_metrics(self):
        self.pixel_conf_matrix_meter.reset()
        self.poly_conf_matrix_meter.reset()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hyperparameters'] = self.hparams
        checkpoint['model_state_dict'] = self.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.hparams.update(checkpoint['hyperparameters'])
        self.load_state_dict(checkpoint['model_state_dict'])

    def configure_models(self) -> None:

        model_type = self.hparams["model"]
        in_channels_S2: int = self.hparams["in_channels_S2"]
        in_channels_S1: int = self.hparams["in_channels_S1"]
        num_classes: int = self.hparams["num_classes"]
        sentinel_type: str = self.hparams["sentinel_type"]
        kernel_size: int = self.hparams["kernel_size"]
        dropout: float = self.hparams["dropout"]

        if model_type == "Unet":
            print("Using FC-EF from fully_conv_models")
            self.model = Unet(
                input_nbr=(in_channels_S2 + in_channels_S1) * 2, 
                label_nbr=num_classes,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        elif model_type == "FCSiamDiff":
            print("Using FCSiamDiff from fully_conv_models")
            if sentinel_type == "both" and in_channels_S2  == 0 and in_channels_S1 == 0:
                raise ValueError("Both S1 and S2 channels cannot be 0")
            if sentinel_type == "S2" and in_channels_S2 == 0:
                raise ValueError("S2 channels cannot be 0")
            if sentinel_type == "S1" and in_channels_S1 == 0:
                raise ValueError("S1 channels cannot be 0")
            self.model = FCSiamDiff(
                input_nbr= in_channels_S2 + in_channels_S1,
                label_nbr=num_classes,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        elif model_type == "FCSiamConc":
            print("Using FCSiamConc from fully_conv_models")
            if sentinel_type == "both" and in_channels_S2  == 0 and in_channels_S1 == 0:
                raise ValueError("Both S1 and S2 channels cannot be 0")
            if sentinel_type == "S2" and in_channels_S2 == 0:
                raise ValueError("S2 channels cannot be 0")
            if sentinel_type == "S1" and in_channels_S1 == 0:
                raise ValueError("S1 channels cannot be 0")
            self.model = FCSiamConc(
                input_nbr = in_channels_S2 + in_channels_S1,
                label_nbr=num_classes,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        elif model_type == "SiamUnet_conc_multi":
            print("Using SiamUnet_conc_multi from fully_conv_models")
            self.model = SiamUnet_conc_multi(
                input_nbr_S2=in_channels_S2,
                input_nbr_S1=in_channels_S1,
                label_nbr=num_classes
            )
        elif model_type == "ChangeformerV8":
            print("Using Changeformer_dual")
            self.model = ChangeFormerV8(
                input_nc = in_channels_S2 + in_channels_S1,
                output_nc = num_classes,
                decoder_softmax = False,
                embed_dim = 32
            )

        elif model_type == "ChangeformerV9":
            print("Using Changeformer_multi")
            self.model = ChangeFormerV9(
                input_nc_s1= in_channels_S1,
                input_nc_s2= in_channels_S2,
                output_nc = num_classes,
                decoder_softmax = False,
                embed_dim = 32
            )

        elif model_type == "ChangeformerV10":
            print("Using Changeformer_single")
            self.model = ChangeFormerV10(
                input_nc = (in_channels_S2 + in_channels_S1) * 2,
                output_nc = num_classes,
                decoder_softmax = False,
                embed_dim = 32
            )


    def configure_losses(self) -> None:

        loss: str = self.hparams["loss"]
        if loss == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1 - self.hparams["deforest_weight"], self.hparams["deforest_weight"]]).cuda()
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports only 'CrossEntropyLoss'"
            )

    def forward(self, x1, x2=None):
        if self.hparams["model"] in ["FCSiamDiff", "FCSiamConc"]:
            return self.model(x1, x2)  
        else:
            return self.model(x1)  