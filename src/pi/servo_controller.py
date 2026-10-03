import logging
class ServoController:
    def __init__(self, cfg, logger=None):
        self.cfg=cfg; self.logger=logger or logging.getLogger("segresmart"); self.servo=None
        if not cfg.get("simulation_mode", True):
            try:
                from gpiozero import Servo
                self.servo=Servo(cfg["gpio_bcm"], min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
            except Exception as exc: raise RuntimeError(f"Servo unavailable; use simulation_mode or fix GPIO permissions: {exc}") from exc
    def _move(self, name, angle):
        self.logger.info("Servo -> %s (%s degrees)",name,angle)
        if self.servo: self.servo.value=(float(angle)-90)/90
    def move_wet(self): self._move("WET",self.cfg["wet_angle"])
    def move_dry(self): self._move("DRY",self.cfg["dry_angle"])
    def move_neutral(self): self._move("NEUTRAL",self.cfg["neutral_angle"])
    def cleanup(self):
        if self.servo: self.servo.detach()
