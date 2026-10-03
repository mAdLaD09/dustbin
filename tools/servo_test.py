import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config
from src.pi.servo_controller import ServoController
cfg=load_config(); servo=ServoController(cfg["servo"]|{"simulation_mode":cfg["system"]["simulation_mode"]})
try:
 while True:
  key=input("1=WET 2=NEUTRAL 3=DRY Q=QUIT: ").strip().lower()
  if key=="1": servo.move_wet()
  elif key=="2": servo.move_neutral()
  elif key=="3": servo.move_dry()
  elif key=="q": break
finally: servo.move_neutral(); servo.cleanup()
