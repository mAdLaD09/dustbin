import argparse, cv2, time
ap=argparse.ArgumentParser(); ap.add_argument("--device",type=int,default=0); ap.add_argument("--save"); a=ap.parse_args(); cap=cv2.VideoCapture(a.device,cv2.CAP_V4L2)
if not cap.isOpened(): raise SystemExit(f"ERROR: could not open camera /dev/video{a.device}. Try: ls /dev/video* and v4l2-ctl --list-devices")
start=time.time(); frames=0
while True:
 ok,frame=cap.read()
 if not ok: continue
 frames+=1; fps=frames/max(time.time()-start,1e-6); cv2.putText(frame,f"{frame.shape[1]}x{frame.shape[0]} FPS {fps:.1f}",(10,30),0,.7,(0,255,0),2); cv2.imshow("camera test",frame)
 if a.save: cv2.imwrite(a.save,frame); a.save=None
 if cv2.waitKey(1)&0xff==ord("q"): break
cap.release(); cv2.destroyAllWindows()
