import argparse, cv2
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument("--device",type=int,default=0); ap.add_argument("--count",type=int,default=400); ap.add_argument("--output",default="datasets/empty"); a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); cap=cv2.VideoCapture(a.device,cv2.CAP_V4L2)
if not cap.isOpened(): raise SystemExit(f"Could not open camera device {a.device}")
i=0
while i<a.count:
 ok,frame=cap.read()
 if not ok: continue
 cv2.putText(frame,f"Empty capture {i+1}/{a.count} - Q quits",(10,30),0,.7,(0,255,0),2); cv2.imshow("empty capture",frame)
 if cv2.waitKey(30)&0xff==ord("q"): break
 cv2.imwrite(str(out/f"empty_{i:04d}.jpg"),frame); i+=1
cap.release(); cv2.destroyAllWindows(); print(f"Captured {i} images. Repeat under realistic lighting conditions.")
