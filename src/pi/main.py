"""Offline Pi application: webcam -> TFLite -> stable decision -> servo."""
import argparse, time, cv2
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config
from src.common.logging_utils import configure_logging, append_prediction
from src.pi.classifier import TFLiteClassifier, load_labels
from src.pi.prediction_filter import PredictionFilter
from src.pi.servo_controller import ServoController
from src.pi.state_machine import SortingStateMachine

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); ap.add_argument("--headless",action="store_true"); args=ap.parse_args(); cfg=load_config(args.config); log=configure_logging(); servo=None; cap=None
    try:
        model=Path(cfg["model"]["path"]); labels=load_labels("models/labels.txt")
        if not model.exists(): raise FileNotFoundError(f"Model missing: {model}")
        classifier=TFLiteClassifier(model,labels,cfg["model"]["image_size"],cfg["model"]["normalization"]); filt=PredictionFilter(cfg["prediction"]["window_size"],cfg["prediction"]["minimum_votes"],cfg["model"]["confidence_threshold"]); servo=ServoController(cfg["servo"]|{"simulation_mode":cfg["system"]["simulation_mode"]},log); servo.move_neutral(); machine=SortingStateMachine(servo,cfg["prediction"]["empty_frames_required"])
        cap=cv2.VideoCapture(cfg["camera"]["index"],cv2.CAP_V4L2); cap.set(cv2.CAP_PROP_FRAME_WIDTH,cfg["camera"]["width"]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg["camera"]["height"])
        if not cap.isOpened(): raise RuntimeError(f"ERROR: USB camera could not be opened at device index {cfg['camera']['index']}.")
        while True:
            ok,frame=cap.read()
            if not ok: log.error("Camera frame read failed"); continue
            x,y,w,h=cfg["camera"]["roi"]; roi=frame[y:y+h,x:x+w]; raw,conf,lat=classifier.predict(roi); stable=filt.update(raw,conf); final=stable.label if stable else ("unknown" if conf<cfg["model"]["confidence_threshold"] else raw); machine.handle(final)
            append_prediction("logs/predictions.csv",{"timestamp":time.time(),"raw_prediction":raw,"final_prediction":final,"confidence":f"{conf:.4f}","inference_latency_ms":f"{lat:.2f}","servo_action":"","state":machine.state.value,"error":""})
            if not args.headless and cfg["system"]["show_preview"]:
                cv2.putText(frame,f"Prediction: {final.upper()} {conf:.1%} {lat:.0f}ms",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2); cv2.imshow("SegreSmart",frame)
                if cv2.waitKey(1)&0xff==ord("q"): break
    except KeyboardInterrupt: pass
    finally:
        if servo:
            try: servo.move_neutral(); servo.cleanup()
            except Exception: pass
        if cap: cap.release()
        cv2.destroyAllWindows()
if __name__=="__main__": main()
