import argparse
import os
import time
from tkinter import Image
from loguru import logger
import numpy as np

import cv2
import torch
import csv
import socket
import tqdm

import pytelicam
from PIL import Image

# add
from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_track
from motpy import Track
from motpy import track_to_string

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_frame",
        action="store_true",
        help="whether to save the inference result of image tracking result",
    )

    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="whether to save the inference result of class and ID tracking result",
    )

    parser.add_argument(
        "--send_data",
        action="store_true",
        help="whether to send the inference result string via TCP",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        
        
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        
        #print("Ratio ",ratio) 
        
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


class MOT:
    def __init__(self):
        self.tracker = MultiObjectTracker(dt=0.1) # 100ms

    def track(self, outputs, ratio):
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            
            outputs = [Detection(box=box[:4] / ratio, score=box[4] * box[5], class_id=box[6]) for box in outputs]
            #print("Outputs' ", len(outputs))
        else:
            outputs = []

        self.tracker.step(detections=outputs)
        tracks = self.tracker.active_tracks()
        
        # print("Tracks' ", tracks)
        # trackはID、bbox、
        # print('MOT tracker tracks %d objects' % len(tracks))
        # print('first track box: %s' % str(tracks[0].box))

        return tracks

class TeliVideoCapture:
    cam_system = None
    instance_count = 0

    # timeout = 5000ms
    timeout = 5000

    def __init__(self, cam_index):
        self.cam_device = None

        if TeliVideoCapture.cam_system == None:
            # It is recommended that the settings of unused interfaces be removed.
            #  (U3v / Gev / GenTL)
            TeliVideoCapture.cam_system = pytelicam.get_camera_system( \
                                              int(pytelicam.CameraType.U3v) | \
                                              int(pytelicam.CameraType.Gev)
                                              )

        cam_num = TeliVideoCapture.cam_system.get_num_of_cameras()
        if cam_index < cam_num:
            self.cam_device = TeliVideoCapture.cam_system.create_device_object(cam_index)
            self.cam_device.open()

            res = self.cam_device.genapi.set_enum_str_value('TriggerMode', 'Off')
            if res != pytelicam.CamApiStatus.Success:
                raise Exception("Can't set TriggerMode.")
            
            status_fps = self.cam_device.genapi.set_feature_value('AcquisitionFrameRate', '30')
            if status_fps != pytelicam.CamApiStatus.Success:
                raise Exception("Can't set Framerate")
            
            status, frame_rate = self.cam_device.genapi.get_feature_value('AcquisitionFrameRate')
            print('status={0}, {1}'.format(status, frame_rate))
            if status != pytelicam.CamApiStatus.Success:
                raise Exception("Can't set Framerate.")

            self.cam_device.cam_stream.open()
            self.cam_device.cam_stream.start()

        TeliVideoCapture.instance_count += 1


    def read(self):
        np_arr = None

        if self.cam_device == None:
            return (False, None)

        with self.cam_device.cam_stream.get_next_image(self.timeout) as image_data:
            if image_data.status != pytelicam.CamApiStatus.Success:
                print('Grab error! status = {0}'.format(image_data.status))
                return (False, None)
            else:
                if image_data.pixel_format == pytelicam.CameraPixelFormat.Mono8:
                    np_arr = image_data.get_ndarray(pytelicam.OutputImageType.Raw)
                else:
                    np_arr = image_data.get_ndarray(pytelicam.OutputImageType.Bgr24)

        return (True, np_arr)


    def release(self):
        if self.cam_device != None:
            if self.cam_device.cam_stream.is_open == True:
                self.cam_device.cam_stream.stop()
                self.cam_device.cam_stream.close()

                if self.cam_device.is_open == True:
                    self.cam_device.close()

            self.cam_device = None

        TeliVideoCapture.instance_count -= 1
        if TeliVideoCapture.instance_count == 0:
            TeliVideoCapture.cam_system.terminate()
            TeliVideoCapture.cam_system = None


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

plc_data = {"pos1":0, "pos2":0, "pos3":0, "pos4":0, "pos5":0, "pos6":0, "pos7":0, 
             "pos8":0, "pos9":0, "pos10":0, "pos11":0, "pos12":0, "pos13":0, 
             "pos14":0, "pos15":0, "pos16":0}

plc_data2 = 0

holdcounter = 20

def write_data2csv():
    
    with open('./csv/plc.csv', 'w', newline="") as f:
        csv_out = csv.writer(f)
         # convert values to list in order to write in columns
        #for v in list(plc_data.values()): # convert values to list in order to write in columns
        csv_out.writerow(list(plc_data.values()))
        csv_out.writerow(str(0))
        
        if( 0 < plc_data["pos1"] ):
            plc_data["pos1"] = plc_data["pos1"] - 1
        if( 0 < plc_data["pos2"] ):
            plc_data["pos2"] = plc_data["pos2"] - 1
        if( 0 < plc_data["pos3"] ):
            plc_data["pos3"] = plc_data["pos3"] - 1
        if( 0 < plc_data["pos4"] ):
            plc_data["pos4"] = plc_data["pos4"] - 1
        if( 0 < plc_data["pos5"] ):
            plc_data["pos5"] = plc_data["pos5"] - 1
        if( 0 < plc_data["pos6"] ):
            plc_data["pos6"] = plc_data["pos6"] - 1
        if( 0 < plc_data["pos7"] ):
            plc_data["pos7"] = plc_data["pos7"] - 1
        if( 0 < plc_data["pos8"] ):
            plc_data["pos8"] = plc_data["pos8"] - 1
        if( 0 < plc_data["pos9"] ):
            plc_data["pos9"] = plc_data["pos9"] - 1
        if( 0 < plc_data["pos10"] ):
            plc_data["pos10"] = plc_data["pos10"] - 1
        if( 0 < plc_data["pos11"] ):
            plc_data["pos11"] = plc_data["pos11"] - 1
        if( 0 < plc_data["pos12"] ):
            plc_data["pos12"] = plc_data["pos12"] - 1
        if( 0 < plc_data["pos13"] ):
            plc_data["pos13"] = plc_data["pos13"] - 1
        if( 0 < plc_data["pos14"] ):
            plc_data["pos14"] = plc_data["pos14"] - 1
        if( 0 < plc_data["pos15"] ):
            plc_data["pos15"] = plc_data["pos15"] - 1
        if( 0 < plc_data["pos16"] ):
            plc_data["pos16"] = plc_data["pos16"] - 1
        
#        for v in list(plc_data.values()):
        # print("PLC_DATA: ", list(plc_data.values()))

HOST = "192.168.0.10"
PORT = 5001
def send_data2socket():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        data_list = list(plc_data.values())
        data_string = "".join(str(i) for i in data_list)

        s.send(data_string.encode())

        data = s.recv(1024)

def imageflow_demo( predictor, vis_folder, current_time, args):
    mot = MOT()
    
    # Open camera that is detected first, in this sample code.
    cap = TeliVideoCapture(0) if args.demo == "video" else args.camid

    frms = 0 
    
    frame_num = 1  
    
    GoodCount1=0
    kizuCount1=0
    moldCount1=0
    kokuCount1=0 

    GoodID1 = []
    kizuID1 = []
    moldID1 = []
    kokuID1 = []

    counted = False # initialize a boolean to check for counted

    output1trcs = []
    # output2trcs = []

    while True:
        print("Press 'ESC' key on the OpenCV window to exit.")
        write_data2csv()
        
        timer = cv2.getTickCount()

        ret_val, frame = cap.read()

        if ret_val:
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            H, W = frame.shape[:2]
            #print("H: ", H)
            
            frms += 1

            PIL_img = Image.fromarray(frame)

            outputs, img_info = predictor.inference(frame)
            
            #result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            result_frame = frame.copy()
            tracks = mot.track(outputs, img_info['ratio'])
            

            # cv2.rectangle(result_frame, (W - 600, 0), (W - 590, H), (255, 0, 0), 2)
           
            # cv2.rectangle(result_frame, (W//2 - 10, 0), (W//2 , H), (255, 255, 0), 2)
            
            cv2.line(result_frame, (1100,0), (1100, H), (255,0,255), 3)

            line_pts = np.linspace(0, H, 17)
            
            #print(list(line_pts))

            for i in list(line_pts):
                cv2.line(result_frame, (1100, int(i)), (W, int(i)), (255,255,255), 1)

            for trc in tracks:
                draw_track(result_frame, trc, thickness=1, text_at_bottom=True, text_verbose=2)

                 # reset csv data to 0s
                Top_right = trc[1][2] 

                # if (trc[1][0]) < W - 590 and (trc[1][0]) >= W - 600:
                if (trc[1][2]) < 1100 and (trc[0],trc[3]) not in output1trcs:

                    all_tracks_1, all_ids_1 = trc[0], trc[3]

                    output1trcs.append((all_tracks_1, all_ids_1))

                    if int(trc[3]) == 0:
                        trk_id, cls = (trc[0], trc[3]) # first detection of object rectangle top left
                        GoodID1.append((trk_id, cls))
                        GoodCount1 += 1

                    if int(trc[3]) == 2:
                        trk_id_k, cls_k = (trc[0], trc[3]) # first detection of defect
                        kizuID1.append((trk_id_k, cls_k))
                        kizuCount1 += 1

                        Half_right = (trc[1][3] - trc[1][1])//2

                    if int(trc[3]) == 1:
                        trk_id_m, cls_m = (trc[0], trc[3]) # first detection of mold
                        
                        moldID1.append((trk_id_m, cls_m))
                        moldCount1 += 1

                        Half_right = (trc[1][3] - trc[1][1])//2
                    
                    if int(trc[3]) == 3:
                        trk_id_kn, cls_kn = (trc[0], trc[3]) # first detection of mold
                        
                        kokuID1.append((trk_id_kn, cls_kn))
                        kokuCount1 += 1

                        Half_right = (trc[1][3] - trc[1][1])//2


                if (trc[1][2]) > 1100 and (trc[0],trc[3]) in output1trcs:

                    current_position_TL = trc[1][2]
                    
                    Half_right = (trc[1][3] - trc[1][1])//2

                    #centrepos_y = (trc[1][1] + trc[1][3])//2

                    if (current_position_TL >= 1100) and (current_position_TL < 1140):
                    #if (current_position_TL == 1130):
                        if int(trc[3]) == 0:
                            
                            #reset_csv_data()
                            cv2.circle(result_frame, (1100, int(trc[1][1])), 50, (255,255,255), thickness=-1)

                        if int(trc[3]) == 2:
                            
                            # boundary_pts = np.linspace(0, 1080, 17)

                            # if (trc[1][1] or trc[1][3]) >= 0 and  (trc[1][1] or trc[1][3]) <= 67.5:
                            #     plc_data["pos1"] = holdcounter
                            #     plc_data["pos2"] = holdcounter
                            #     plc_data["pos3"] = holdcounter
                            if (trc[1][1] or trc[1][3]) >= 0 and  (trc[1][1] or trc[1][3]) <= 135:
                                plc_data["pos1"] = holdcounter
                                plc_data["pos2"] = holdcounter
                                plc_data["pos3"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 135 and  (trc[1][1] or trc[1][3]) <= 202.5:
                                plc_data["pos2"] = holdcounter
                                plc_data["pos3"] = holdcounter
                                plc_data["pos4"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 202.5 and  (trc[1][1] or trc[1][3]) <= 270:
                                plc_data["pos3"] = holdcounter
                                plc_data["pos4"] = holdcounter
                                plc_data["pos5"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 270 and  (trc[1][1] or trc[1][3]) <= 337.5:
                                plc_data["pos4"] = holdcounter
                                plc_data["pos5"] = holdcounter
                                plc_data["pos6"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 337.5 and  (trc[1][1] or trc[1][3]) <= 405:
                                plc_data["pos5"] = holdcounter
                                plc_data["pos6"] = holdcounter
                                plc_data["pos7"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 405 and  (trc[1][1] or trc[1][3]) <= 472.5:
                                plc_data["pos6"] = holdcounter
                                plc_data["pos7"] = holdcounter
                                plc_data["pos8"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 472.5 and  (trc[1][1] or trc[1][3]) <= 540:
                                plc_data["pos7"] = holdcounter
                                plc_data["pos8"] = holdcounter
                                plc_data["pos9"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 540 and  (trc[1][1] or trc[1][3]) <= 607.5:
                                plc_data["pos8"] = holdcounter
                                plc_data["pos9"] = holdcounter
                                plc_data["pos10"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 607.5 and  (trc[1][1] or trc[1][3]) <= 675:
                                plc_data["pos9"] = holdcounter
                                plc_data["pos10"] = holdcounter
                                plc_data["pos11"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 675 and  (trc[1][1] or trc[1][3]) <= 742.5:
                                plc_data["pos10"] = holdcounter
                                plc_data["pos11"] = holdcounter
                                plc_data["pos12"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 742.5 and  (trc[1][1] or trc[1][3]) <= 810:
                                plc_data["pos11"] = holdcounter
                                plc_data["pos12"] = holdcounter
                                plc_data["pos13"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 810 and  (trc[1][1] or trc[1][3]) <= 877.5:
                                plc_data["pos12"] = holdcounter
                                plc_data["pos13"] = holdcounter
                                plc_data["pos14"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 877.5 and  (trc[1][1] or trc[1][3]) <= 945:
                                plc_data["pos13"] = holdcounter
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 945 and  (trc[1][1] or trc[1][3]) <= 1012.5:
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                                plc_data["pos16"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 1012.5 and  (trc[1][1] or trc[1][3]) <= 1080:
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                                plc_data["pos16"] = holdcounter
                            
                                
                            cv2.circle(result_frame, (1100, int(trc[1][1])), 30, (255,0,0), thickness=-1)

                            if args.send_data:
                                send_data2socket()


                        if int(trc[3]) == 1:

                            # if (trc[1][1] or trc[1][3]) >= 0 and  (trc[1][1] or trc[1][3]) <= 67.5:
                            #     plc_data["pos1"] = holdcounter
                            #     plc_data["pos2"] = holdcounter
                            #     plc_data["pos3"] = holdcounter
                            if (trc[1][1] or trc[1][3]) >= 0 and  (trc[1][1] or trc[1][3]) <= 135:
                                plc_data["pos1"] = holdcounter
                                plc_data["pos2"] = holdcounter
                                plc_data["pos3"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 135 and  (trc[1][1] or trc[1][3]) <= 202.5:
                                plc_data["pos2"] = holdcounter
                                plc_data["pos3"] = holdcounter
                                plc_data["pos4"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 202.5 and  (trc[1][1] or trc[1][3]) <= 270:
                                plc_data["pos3"] = holdcounter
                                plc_data["pos4"] = holdcounter
                                plc_data["pos5"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 270 and  (trc[1][1] or trc[1][3]) <= 337.5:
                                plc_data["pos4"] = holdcounter
                                plc_data["pos5"] = holdcounter
                                plc_data["pos6"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 337.5 and  (trc[1][1] or trc[1][3]) <= 405:
                                plc_data["pos5"] = holdcounter
                                plc_data["pos6"] = holdcounter
                                plc_data["pos7"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 405 and  (trc[1][1] or trc[1][3]) <= 472.5:
                                plc_data["pos6"] = holdcounter
                                plc_data["pos7"] = holdcounter
                                plc_data["pos8"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 472.5 and  (trc[1][1] or trc[1][3]) <= 540:
                                plc_data["pos7"] = holdcounter
                                plc_data["pos8"] = holdcounter
                                plc_data["pos9"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 540 and  (trc[1][1] or trc[1][3]) <= 607.5:
                                plc_data["pos8"] = holdcounter
                                plc_data["pos9"] = holdcounter
                                plc_data["pos10"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 607.5 and  (trc[1][1] or trc[1][3]) <= 675:
                                plc_data["pos9"] = holdcounter
                                plc_data["pos10"] = holdcounter
                                plc_data["pos11"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 675 and  (trc[1][1] or trc[1][3]) <= 742.5:
                                plc_data["pos10"] = holdcounter
                                plc_data["pos11"] = holdcounter
                                plc_data["pos12"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 742.5 and  (trc[1][1] or trc[1][3]) <= 810:
                                plc_data["pos11"] = holdcounter
                                plc_data["pos12"] = holdcounter
                                plc_data["pos13"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 810 and  (trc[1][1] or trc[1][3]) <= 877.5:
                                plc_data["pos12"] = holdcounter
                                plc_data["pos13"] = holdcounter
                                plc_data["pos14"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 877.5 and  (trc[1][1] or trc[1][3]) <= 945:
                                plc_data["pos13"] = holdcounter
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 945 and  (trc[1][1] or trc[1][3]) <= 1012.5:
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                                plc_data["pos16"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 1012.5 and  (trc[1][1] or trc[1][3]) <= 1080:
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                                plc_data["pos16"] = holdcounter
                            
                            cv2.circle(result_frame, (1100, int(trc[1][1])), 30, (0,0,255), thickness=-1)


                            if args.send_data:
                                send_data2socket()

                        if int(trc[3]) == 3:

                            # if (trc[1][1] or trc[1][3]) >= 0 and  (trc[1][1] or trc[1][3]) <= 67.5:
                            #     plc_data["pos1"] = holdcounter
                            #     plc_data["pos2"] = holdcounter
                            #     plc_data["pos3"] = holdcounter
                            if (trc[1][1] or trc[1][3]) >= 0 and  (trc[1][1] or trc[1][3]) <= 135:
                                plc_data["pos1"] = holdcounter
                                plc_data["pos2"] = holdcounter
                                plc_data["pos3"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 135 and  (trc[1][1] or trc[1][3]) <= 202.5:
                                plc_data["pos2"] = holdcounter
                                plc_data["pos3"] = holdcounter
                                plc_data["pos4"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 202.5 and  (trc[1][1] or trc[1][3]) <= 270:
                                plc_data["pos3"] = holdcounter
                                plc_data["pos4"] = holdcounter
                                plc_data["pos5"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 270 and  (trc[1][1] or trc[1][3]) <= 337.5:
                                plc_data["pos4"] = holdcounter
                                plc_data["pos5"] = holdcounter
                                plc_data["pos6"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 337.5 and  (trc[1][1] or trc[1][3]) <= 405:
                                plc_data["pos5"] = holdcounter
                                plc_data["pos6"] = holdcounter
                                plc_data["pos7"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 405 and  (trc[1][1] or trc[1][3]) <= 472.5:
                                plc_data["pos6"] = holdcounter
                                plc_data["pos7"] = holdcounter
                                plc_data["pos8"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 472.5 and  (trc[1][1] or trc[1][3]) <= 540:
                                plc_data["pos7"] = holdcounter
                                plc_data["pos8"] = holdcounter
                                plc_data["pos9"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 540 and  (trc[1][1] or trc[1][3]) <= 607.5:
                                plc_data["pos8"] = holdcounter
                                plc_data["pos9"] = holdcounter
                                plc_data["pos10"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 607.5 and  (trc[1][1] or trc[1][3]) <= 675:
                                plc_data["pos9"] = holdcounter
                                plc_data["pos10"] = holdcounter
                                plc_data["pos11"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 675 and  (trc[1][1] or trc[1][3]) <= 742.5:
                                plc_data["pos10"] = holdcounter
                                plc_data["pos11"] = holdcounter
                                plc_data["pos12"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 742.5 and  (trc[1][1] or trc[1][3]) <= 810:
                                plc_data["pos11"] = holdcounter
                                plc_data["pos12"] = holdcounter
                                plc_data["pos13"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 810 and  (trc[1][1] or trc[1][3]) <= 877.5:
                                plc_data["pos12"] = holdcounter
                                plc_data["pos13"] = holdcounter
                                plc_data["pos14"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 877.5 and  (trc[1][1] or trc[1][3]) <= 945:
                                plc_data["pos13"] = holdcounter
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 945 and  (trc[1][1] or trc[1][3]) <= 1012.5:
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                                plc_data["pos16"] = holdcounter
                            elif (trc[1][1] or trc[1][3]) >= 1012.5 and  (trc[1][1] or trc[1][3]) <= 1080:
                                plc_data["pos14"] = holdcounter
                                plc_data["pos15"] = holdcounter
                                plc_data["pos16"] = holdcounter

                            cv2.circle(result_frame, (1100, int(trc[1][1])), 30, (255,255,0), thickness=-1)
    
                            if args.send_data:
                                send_data2socket()
           
            info = [("Good1", GoodCount1), ("Defect1", kizuCount1), ("Mold1", moldCount1), ("koku1", kokuCount1)]   
 
            # loop over the info tuples and draw them on the frame
            for i, (k,v) in enumerate(info):
               text = "{}:{}".format(k,v)
            #    cv2.putText(result_frame, text, (H - ((i * 150) + 40), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)  
            
            if args.save_frame:
                cv2.imwrite("frame%d.jpg" %frame_num, result_frame)
                frame_num +=1
            
            

                #with open('./csv/output1.csv', 'w', newline="") as f:
                    #csv_out = csv.writer(f)
                    #csv_out.writerow(['TrackID', 'classID'])
                    #for row in GoodID1:
                        #csv_out.writerow(row)
                    #for row1 in kizuID1:
                        #csv_out.writerow(row1)
                    #for row2 in moldID1:
                        #csv_out.writerow(row2)
                    #for row3 in kokuID1:
                        #csv_out.writerow(row3)

                #with open('./csv/AlltracksROI_1.csv', 'w', newline="") as f:
                    #csv_out = csv.writer(f)
                    #csv_out.writerow(['TrackID', 'classID'])
                    #for row in output1trcs:
                        #csv_out.writerow(row)
                
                  
            if args.save_result:
                save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
                # print(os.path.splitext(args.path.split("/")[-1])[0] + '.mp4')
                os.makedirs(save_folder, exist_ok=True)
                if args.demo == "video":
                    # save_path = os.path.join(save_folder, os.path.splitext(args.path.split("/")[-1])[0] + '.mp4')
                    save_path = os.path.join(save_folder,  'Result.mp4')
                    logger.info(f"video save_path is {save_path}")
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(W), int(H)))
                    
                    vid_writer.write(result_frame)
                    
                    print("Saving video to {}".format(save_path))
                    
                else:
                    save_path = os.path.join(save_folder, "camera.mp4")
                    logger.info(f"video save_path is {save_path}")
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(W), int(H)))
                    
                    vid_writer.write(result_frame)
                    print("Saving video to {}".format(save_path))
            else:
                "Display"
                cv2.imshow('frame', result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
        else:
            "Breaking>>>>>>"
            break
    #print("Frame #: ", frms)
    

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "test_res")
        os.makedirs(vis_folder, exist_ok=True)
    
    #if args.save_frame:
    #    frame_folder = os.path.join(file_name, "vis_res")
    #    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)


