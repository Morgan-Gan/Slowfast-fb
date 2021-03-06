#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import queue
import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import slowfast.utils.checkpoint as cu
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.utils import loggings
from slowfast.visualization.utils import process_cv2_inputs

import numpy as np

logger = loggings.get_logger(__name__)


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # Build the video model and print model statistics.
        self.model = build_model(cfg, gpu_id=gpu_id)
        self.model.eval()
        self.cfg = cfg

        if cfg.DETECTION.ENABLE:
            self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)

        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    # __call__：方法把Predictor变成可调用对象，并可传入tast参数
    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        # * ------ 1. first stage : starting detection ----------------------*/
        if self.cfg.DETECTION.ENABLE:
            task = self.object_detector(task)

       # * ------ 2. Second stage : starting recognition ----------------------*/
        frames, bboxes = task.frames, task.bboxes

        ################################################################################################################
        from slowfast.datasets.utils import pack_pathway_output, tensor_normalize
        from torchvision import transforms
        from PIL import Image
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
            ]

        inputs1 = []
        inputs0 = []
        cv2_transform.lineSpace(0, 63, 32, frames, inputs1)
        cv2_transform.lineSpace(0, 31, 8, inputs1, inputs0)

        inputs0 = [
            cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
            for frame in inputs0
        ]
        inputs1 = [
            cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
            for frame in inputs1
        ]

        inputs0 = torch.from_numpy(np.array(inputs0)).float() / 255
        inputs1 = torch.from_numpy(np.array(inputs1)).float() / 255
        inputs0 = tensor_normalize(
            inputs0, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        inputs1 = tensor_normalize(
            inputs1, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # T H W C -> C T H W.
        inputs0 = inputs0.permute(3, 0, 1, 2)
        inputs1 = inputs1.permute(3, 0, 1, 2)
        inputs0 = inputs0.unsqueeze(0)
        inputs1 = inputs1.unsqueeze(0)
        inputs = [inputs0, inputs1]
        ###############################################################################################################

        if bboxes is not None:
            bboxes = cv2_transform.scale_boxes(
                self.cfg.DATA.TEST_CROP_SIZE,
                bboxes,
                task.img_height,
                task.img_width,
            )
        # if self.cfg.DEMO.INPUT_FORMAT == "BGR":
        #     frames = [
        #         cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
        #     ]

        # frames = [
        #     cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
        #     for frame in frames
        # ]

        # change frames to slowfast inputs
        # inputs = process_cv2_inputs(frames, self.cfg)
        # add person cls to bbox
        if bboxes is not None:
            index_pad = torch.full(
                size=(bboxes.shape[0], 1),
                fill_value=float(0),
                device=bboxes.device,
            )

            # Pad frame index for each box.
            bboxes = torch.cat([index_pad, bboxes], axis=1)
        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )
        if self.cfg.DETECTION.ENABLE and not bboxes.shape[0]:
            preds = torch.tensor([])
        else:
            # change    {1,3,8,224,224]->[8,3,224,224]
            bboxes = bboxes.unsqueeze(0).unsqueeze(0)
            inputs[0] = inputs[0].squeeze(0).permute(1, 0, 2, 3)
            inputs[1] = inputs[1].squeeze(0).permute(1, 0, 2, 3)
            ##########################################################
            import numpy
            numpy.set_printoptions(suppress=True)

            # import scipy.io as io
            # inputs0 = inputs[0].squeeze(0).permute(
            #     1, 0, 2, 3)[0].permute(1, 2, 0).data.cpu().numpy()
            # cv2.imwrite("1.jpg", np.array(
            #     inputs0*255, dtype=np.float32))  # dtype=np.uint8
            # print(inputs0)
            # numpy.save("input0.npy", inputs0)
            # result0 = numpy.array(inputs0.reshape(-1, 1))
            # numpy.savetxt("result0.txt", result0)
            # io.savemat("save.mat", {"result0": result0})

#######################  save .txt file ############################
            # result0 = numpy.array(
            #     inputs[0].cpu().reshape(-1, 1)).astype(np.float32)
            # # result0 = result0.astype('float')
            # # for i in range(10):
            # #     print(result0[i])
            # # exit(0)
            # result0.astype('float32').tofile("input0.txt")
            # result1 = numpy.array(
            #     inputs[1].cpu().reshape(-1, 1)).astype(np.float32)
            # result1.astype('float32').tofile("input1.txt")
            # result0 = numpy.array(
            #     bboxes.cpu().reshape(-1, 1)).astype(np.float32)
            # result0.astype('float32').tofile("input2.txt")

##################################### save .npy file ###################
            # numpy.save("input0.npy", inputs[0].cpu().numpy())
            # numpy.save("input1.npy", inputs[1].cpu().numpy())
            # numpy.save("input2.npy", bboxes.cpu().numpy())
            # input0 = torch.from_numpy(np.load("input0.npy")).cuda()
            # input1 = torch.from_numpy(np.load("input1.npy")).cuda()
            # input2 = torch.from_numpy(np.load("input2.npy")).cuda()
            ##########################################################
            preds = self.model(inputs, bboxes)
            # preds = self.model([input0, input1], input2)

            # result_pred = numpy.array(preds.detach().cpu().reshape(-1, 1))
            # numpy.savetxt("result_preds.txt", result_pred)
            print(preds)
            exit(0)
            #*****************************   open with video test ##########################
            bboxes = bboxes.squeeze(0).squeeze(0)  # change[1,1,3,5] -->[3,5]
            #*****************************   open with video test end ##########################

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()
            if bboxes is not None:
                bboxes = bboxes.detach().cpu()

        preds = preds.detach()
        task.add_action_preds(preds)
        if bboxes is not None:
            task.add_bboxes(bboxes[:, 1:])

        return task


class ActionPredictor:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis=None, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        self.predictor = Predictor(cfg=cfg, gpu_id=gpu_id)
        self.async_vis = async_vis

    def put(self, task):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        task = self.predictor(
            task)                           # 返回Predictor中__call__动作检测结果
        self.async_vis.get_indices_ls.append(task.id)
        self.async_vis.put(task)

    def get(self):
        """
        Get the visualized clips if any.
        """
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")

        return task


class Detectron2Predictor:
    """
    Wrapper around Detectron2 to return the required predicted bounding boxes
    as a ndarray.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG)
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        if cfg.NUM_GPUS and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.cfg.MODEL.DEVICE = (
            "cuda:{}".format(gpu_id) if cfg.NUM_GPUS > 0 else "cpu"
        )

        logger.info("Initialized Detectron2 Object Detection Model.")

        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, task):
        """
        Return bounding boxes predictions as a tensor.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        middle_frame = task.frames[len(task.frames) // 2]
        outputs = self.predictor(middle_frame)
        # Get only human instances
        mask = outputs["instances"].pred_classes == 0
        pred_boxes = outputs["instances"].pred_boxes.tensor[mask]
        task.add_bboxes(pred_boxes)

        return task
