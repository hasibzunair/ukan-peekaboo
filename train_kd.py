# Code for Peekaboo
# Author: Hasib Zunair
# Modified from https://github.com/valeoai/FOUND

"""Training code for Peekaboo"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from tqdm import tqdm

from model import PeekabooModel
from evaluation.saliency import evaluate_saliency
from misc import (
    batch_apply_bilateral_solver,
    set_seed,
    load_config,
    Logger,
    DistillationLoss,
)

from datasets.datasets import build_dataset
from ukan import UKAN


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Training of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Exp name.")
    parser.add_argument(
        "--log-dir", type=str, default="outputs", help="Logging and output directory."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root directories of training and evaluation datasets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/peekaboo_DUTS-TR.yaml",
        help="Path of config file.",
    )
    parser.add_argument(
        "--save-model-freq", type=int, default=250, help="Frequency of model saving."
    )
    parser.add_argument(
        "--visualization-freq",
        type=int,
        default=10,
        help="Frequency of prediction visualization in tensorboard.",
    )

    args = parser.parse_args()
    return args


def train_model(
    teacher_model,
    student_model,
    config,
    dataset,
    dataset_dir,
    visualize_freq=10,
    save_model_freq=500,
    tensorboard_log_dir=None,
):

    # Diverse
    print(f"Data will be saved in {tensorboard_log_dir}")
    save_dir = tensorboard_log_dir
    if tensorboard_log_dir is not None:
        # Logging
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(tensorboard_log_dir)

    # Deconvolution, train only the decoder
    sigmoid = nn.Sigmoid()
    student_model.train()
    student_model.to("cuda")

    ################################################################################
    #                                                                              #
    #                      Setup loss, optimizer and scheduler                     #
    #                                                                              #
    ################################################################################

    criterion = nn.BCEWithLogitsLoss()
    distillation_loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)

    param_groups = []
    for name, param in student_model.named_parameters():
        # print(name, "=>", param.shape)
        if "layer" in name.lower() and "fc" in name.lower():  # higher lr for kan layers
            param_groups.append({"params": param, "lr": 1e-2, "weight_decay": 1e-4})
        else:
            param_groups.append({"params": param, "lr": 1e-4, "weight_decay": 1e-4})

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training["nb_epochs"], eta_min=1e-5
    )

    # Peekaboo
    # optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config.training["lr0"])
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=config.training["step_lr_size"],
    #     gamma=config.training["step_lr_gamma"],
    # )

    ################################################################################
    #                                                                              #
    #                                Dataset                                       #
    #                                                                              #
    ################################################################################

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training["batch_size"], shuffle=True, num_workers=2
    )

    ################################################################################
    #                                                                              #
    #                                Training loop                                 #
    #                                                                              #
    ################################################################################

    n_iter = 0
    for epoch in range(config.training["nb_epochs"]):
        running_loss = 0.0
        tbar = tqdm(enumerate(trainloader, 0), leave=None)
        for i, data in tbar:

            # Get the inputs
            inputs, masked_inputs, _, input_nonorm, masked_input_nonorm, _, _ = data

            # Inputs and masked inputs
            inputs = inputs.to("cuda")
            masked_inputs = masked_inputs.to("cuda")

            # zero the parameter gradients
            optimizer.zero_grad()

            ################################################################################
            #                                                                              #
            #                                Student Model                                 #
            #                                                                              #
            ################################################################################

            # Get predictions
            preds = student_model(inputs)
            # Binarization
            preds_mask = (sigmoid(preds.detach()) > 0.5).float()
            # Apply bilateral solver
            preds_mask_bs, _ = batch_apply_bilateral_solver(data, preds_mask.detach())
            # Flatten
            flat_preds = preds.permute(0, 2, 3, 1).reshape(-1, 1)

            ################################################################################
            #                                                                              #
            #                                Teacher Model                                 #
            #                                                                              #
            ################################################################################

            with torch.no_grad():
                teacher_preds = teacher_model(inputs)

            # Binarization
            teacher_preds_mask = (sigmoid(teacher_preds.detach()) > 0.5).float()
            # Apply bilateral solver
            teacher_preds_mask_bs, _ = batch_apply_bilateral_solver(
                data, teacher_preds_mask.detach()
            )
            # Flatten
            flat_teacher_preds = teacher_preds.permute(0, 2, 3, 1).reshape(-1, 1)

            #### Compute loss ####
            dist_loss = distillation_loss_fn(
                flat_preds,
                flat_teacher_preds,
                teacher_preds_mask_bs.reshape(-1).float()[:, None],
            )
            loss = dist_loss
            writer.add_scalar("Loss/L_dist", dist_loss, n_iter)

            #### Compute loss between soft masks of student and teachers binarized versions ####
            self_loss = criterion(
                flat_preds, teacher_preds_mask.reshape(-1).float()[:, None]
            )

            self_loss = self_loss * 2
            loss += self_loss
            writer.add_scalar("Loss/L_regularization", self_loss, n_iter)

            ################################################################################
            #                                                                              #
            #                       Update weights and scheduler step                      #
            #                                                                              #
            ################################################################################

            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/total_loss", loss, n_iter)
            writer.add_scalar("params/lr", optimizer.param_groups[0]["lr"], n_iter)
            scheduler.step()

            ################################################################################
            #                                                                              #
            #                       Visualize predictions and show stats                   #
            #                                                                              #
            ################################################################################

            # Visualize predictions in tensorboard
            if n_iter % visualize_freq == 0:
                # images and predictions
                grid = torchvision.utils.make_grid(input_nonorm[:5])
                writer.add_image("training/images", grid, n_iter)
                p_grid = torchvision.utils.make_grid(preds_mask[:5])
                writer.add_image("training/preds", p_grid, n_iter)

                # # masked images and predictions
                # m_grid = torchvision.utils.make_grid(masked_input_nonorm[:5])
                # writer.add_image("training/masked_images", m_grid, n_iter)
                # mp_grid = torchvision.utils.make_grid(preds_mask_mfp[:5])
                # writer.add_image("training/masked_preds", mp_grid, n_iter)
            # Statistics
            running_loss += loss.item()
            tbar.set_description(
                f"{dataset.name}| train | iter {n_iter} | loss: ({running_loss / (i + 1):.3f}) "
            )

            ################################################################################
            #                                                                              #
            #                           Save model and evaluate                            #
            #                                                                              #
            ################################################################################

            # Save model
            if n_iter % save_model_freq == 0 and n_iter > 0:
                # model.decoder_save_weights(save_dir, n_iter)
                torch.save(student_model.state_dict(), f"{save_dir}/model.pth")

            # Evaluation
            if n_iter % config.evaluation["freq"] == 0 and n_iter > 0:
                for dataset_eval_name in config.evaluation["datasets"]:
                    val_dataset = build_dataset(
                        root_dir=dataset_dir,
                        dataset_name=dataset_eval_name,
                        for_eval=True,
                        dataset_set=None,
                    )
                    evaluate_saliency(
                        val_dataset, model=student_model, n_iter=n_iter, writer=writer
                    )

            if n_iter == config.training["max_iter"]:
                # model.decoder_save_weights(save_dir, n_iter)
                torch.save(student_model.state_dict(), f"{save_dir}/model.pth")
                print("\n----" "\nTraining done.")
                writer.close()
                return student_model

            n_iter += 1

        print(f"##### Number of epoch is {epoch} and n_iter is {n_iter} #####")

    # Save model
    # model.decoder_save_weights(save_dir, n_iter)
    torch.save(student_model.state_dict(), f"{save_dir}/model.pth")
    print("\n----" "\nTraining done.")
    writer.close()
    return student_model


def main():

    ########## Get arguments ##########

    args = get_argparser()

    ########## Setup ##########

    # Load config yaml file
    config, config_ = load_config(args.config)

    # Experiment name
    exp_name = "{}-{}{}".format(
        config.training["dataset"], config.model["arch"], config.model["patch_size"]
    )

    if args.exp_name is not None:
        exp_name = f"{args.exp_name}-{exp_name}"

    # Log dir
    output_dir = os.path.join(args.log_dir, exp_name)

    # Logging
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save config
    with open(f"{output_dir}/config.json", "w") as f:
        print(f"Config saved in {output_dir}/config.json.")
        json.dump(args.__dict__, f)

    # Save output of terminal in log file
    sys.stdout = Logger(os.path.join(output_dir, "log_train.txt"))
    arguments = str(args).split(", ")
    print("=========================\nConfigs:{}\n=========================")
    for i in range(len(arguments)):
        print(arguments[i])
    print(
        "Hyperparameters from config file: "
        + ", ".join(f"{k}={v}" for k, v in config_.items())
    )
    print("=========================")

    ########## Reproducibility ##########

    set_seed(config.training["seed"])

    ########## Build training set ##########

    dataset = build_dataset(
        root_dir=args.dataset_dir,
        dataset_name=config.training["dataset"],
        dataset_set=config.training["dataset_set"],
        config=config,
        for_eval=False,
    )

    dataset_set = config.training["dataset_set"]
    str_set = dataset_set if dataset_set is not None else ""
    print(f"\nBuilding dataset {dataset.name}{str_set} of {len(dataset)}")

    ########## Define Peekaboo Teacher ##########

    teacher_model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    # Load weights
    teacher_model.decoder_load_weights(
        "data/weights/peekaboo_decoder_weights_niter500.pt"
    )
    teacher_model.eval()
    teacher_model.to("cuda")

    ########## Define UKAN Student ##########

    student_model = UKAN(num_classes=1)

    ########## Training and evaluation ##########

    print(f"\nStarted training on {dataset.name} [tensorboard dir: {output_dir}]")
    student_model = train_model(
        teacher_model=teacher_model,
        student_model=student_model,
        config=config,
        dataset=dataset,
        dataset_dir=args.dataset_dir,
        tensorboard_log_dir=output_dir,
        visualize_freq=args.visualization_freq,
        save_model_freq=args.save_model_freq,
    )
    print(f"\nTraining done, Peekaboo model saved in {output_dir}.")


if __name__ == "__main__":
    main()
