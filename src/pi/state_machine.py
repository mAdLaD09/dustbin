from enum import Enum
class State(str,Enum): STARTUP="STARTUP"; IDLE="IDLE"; ACTUATING="ACTUATING"; WAIT_FOR_REMOVAL="WAIT_FOR_REMOVAL"; COOLDOWN="COOLDOWN"; ERROR="ERROR"; SHUTDOWN="SHUTDOWN"
class SortingStateMachine:
    def __init__(self, servo, empty_frames_required=8): self.servo=servo; self.empty_required=empty_frames_required; self.empty_count=0; self.state=State.IDLE
    def handle(self, prediction):
        if self.state in (State.IDLE,State.COOLDOWN) and prediction in ("wet","dry"):
            getattr(self.servo,"move_"+prediction)(); self.state=State.WAIT_FOR_REMOVAL; self.empty_count=0; return True
        if self.state==State.WAIT_FOR_REMOVAL:
            if prediction in ("empty","unknown",None): self.empty_count+=1
            else: self.empty_count=0
            if self.empty_count>=self.empty_required: self.servo.move_neutral(); self.state=State.IDLE
        return False
    def shutdown(self): self.servo.move_neutral(); self.state=State.SHUTDOWN
