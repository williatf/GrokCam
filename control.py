import wiringpi as wiringpi
import time

class tcControl:
    def __init__(self):
        wiringpi.wiringPiSetup()
        self.MCP23S17_ADDR = 0x20
        self.MCP23S17_IODIRA = 0x00
        self.MCP23S17_IODIRB = 0x01
        self.MCP23S17_GPIOA = 0x12
        self.MCP23S17_GPIOB = 0x13
        wiringpi.mcp23s17Setup(100, 0, self.MCP23S17_ADDR)
        wiringpi.pinMode(100, 1)  # Outputs for stepper, takeup, LED, shutter
        wiringpi.pinMode(101, 1)
        wiringpi.pinMode(102, 1)
        wiringpi.pinMode(103, 1)
        wiringpi.pinMode(104, 1)
        wiringpi.pinMode(105, 1)
        wiringpi.pinMode(106, 1)
        wiringpi.pinMode(107, 1)
        wiringpi.pinMode(108, 1)
        wiringpi.pinMode(109, 1)
        wiringpi.pinMode(110, 1)
        self.STEPPER_PINS = [101, 102, 100]  # step, dir, enable
        self.STEPPER_PINS2 = [104, 103, 105]
        self.TAKEUP_PINS = [106, 107]
        self.LED_PIN = 108
        self.SHUTTER_PINS = [109, 110]
        self.steps_taken = 0
        wiringpi.digitalWrite(self.LED_PIN, 0)
        wiringpi.digitalWrite(self.STEPPER_PINS[2], 0) #enable
        wiringpi.digitalWrite(self.STEPPER_PINS2[2], 0) #enable
        wiringpi.digitalWrite(self.STEPPER_PINS[1], 1) #direction forward
        wiringpi.digitalWrite(self.STEPPER_PINS2[1], 1) #direction forward

        self.PUSHER_RATIO = 0.98 # push ~2% less than pull

    def light_on(self):
        wiringpi.digitalWrite(self.LED_PIN, 1)

    def light_off(self):
        wiringpi.digitalWrite(self.LED_PIN, 0)

    def steps_forward(self, steps=1):
        # Puller is master, always moves
        # Pusher moves according to PUSHER_RATIO
        pusher_counter = 0.0

        for _ in range(steps):
            # Decide if pusher should move this step
            pusher_counter += self.PUSHER_RATIO
            pusher_step = pusher_counter >= 1.0

            # --- STEP HIGH ---
            wiringpi.digitalWrite(self.STEPPER_PINS2[0], 1)  # puller step
            if pusher_step:
                wiringpi.digitalWrite(self.STEPPER_PINS[0], 1)  # pusher step
                pusher_counter -= 1.0

            time.sleep(0.001)

            # --- STEP LOW ---
            wiringpi.digitalWrite(self.STEPPER_PINS2[0], 0)
            if pusher_step:
                wiringpi.digitalWrite(self.STEPPER_PINS[0], 0)

            time.sleep(0.001)

            self.steps_taken += 1
            if self.steps_taken >= 550:
                self.takeup_pulse()
                self.steps_taken = 0

    def steps_back(self, steps=1):
        wiringpi.digitalWrite(self.STEPPER_PINS[1], 0) #direction backwards
        wiringpi.digitalWrite(self.STEPPER_PINS2[1], 0)
        for _ in range(steps):
            wiringpi.digitalWrite(self.STEPPER_PINS[0], 1)
            wiringpi.digitalWrite(self.STEPPER_PINS2[0], 1)
            time.sleep(0.002)
            wiringpi.digitalWrite(self.STEPPER_PINS[0], 0)
            wiringpi.digitalWrite(self.STEPPER_PINS2[0], 0)
            time.sleep(0.002)
        wiringpi.digitalWrite(self.STEPPER_PINS[1], 1) #direction back to foward
        wiringpi.digitalWrite(self.STEPPER_PINS2[1], 1)

    def takeup_pulse(self):
        for pin in self.TAKEUP_PINS:
            wiringpi.digitalWrite(pin, 1)
            time.sleep(0.1)
            wiringpi.digitalWrite(pin, 0)

    def clean_up(self):
        self.light_off()
        wiringpi.digitalWrite(self.STEPPER_PINS[2], 0)
