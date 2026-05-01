from __future__ import division
import lgpio
import time

class tcControl:
    pin_base = 100
    m1_en_pin = pin_base + 0
    m1_step_pin = pin_base + 1
    m1_dir_pin = pin_base + 2
    m2_en_pin = pin_base + 5
    m2_step_pin = pin_base + 4
    m2_dir_pin = pin_base + 3
    reel1_pin = pin_base + 6
    reel2_pin = pin_base + 7
    led_pin = pin_base + 8
    shutter_pin = pin_base + 9
    focus_pin = pin_base + 10
    gpio1_pin = pin_base + 11
    gpio2_pin = pin_base + 12
    gpio3_pin = pin_base + 13
    gpio4_pin = pin_base + 14
    gpio5_pin = pin_base + 15

    feed_interval = 5000
    take_up_steps = 5000
    feed_counter = 0
    take_up_counter = 0
    feed_pulse_delay = 0.1
    takeup_pulse_delay = 0.2
    tension_steps = 50
    step_counter = 0
    advance_settle_delay = 0.01
    post_takeup_settle_delay = 0.01

    def __init__(self):
        self.h_spi = lgpio.spi_open(0)  # SPI channel 0
        self.MCP23S17_ADDR = 0x20
        self.MCP23S17_IODIRA = 0x00
        self.MCP23S17_IODIRB = 0x01
        self.MCP23S17_GPIOA = 0x12
        self.MCP23S17_GPIOB = 0x13
        # Initialize MCP23S17: Ports A/B as outputs
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.MCP23S17_ADDR << 1), self.MCP23S17_IODIRA, 0x00]))
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.MCP23S17_ADDR << 1), self.MCP23S17_IODIRB, 0x00]))
        # Stepper motors
        self.m1 = stepperMotor(self.m1_step_pin, self.m1_dir_pin, self.m1_en_pin, self.h_spi, self.MCP23S17_ADDR)
        self.m2 = stepperMotor(self.m2_step_pin, self.m2_dir_pin, self.m2_en_pin, self.h_spi, self.MCP23S17_ADDR)
        self.reel1 = reelMotor(self.reel1_pin, self.h_spi, self.MCP23S17_ADDR)
        self.reel2 = reelMotor(self.reel2_pin, self.h_spi, self.MCP23S17_ADDR)
        self.led = ledControl(self.led_pin, self.h_spi, self.MCP23S17_ADDR)
        self.shutter_release = shutterRelease(self.focus_pin, self.shutter_pin, self.h_spi, self.MCP23S17_ADDR)
        self.m1.on()
        self.m2.on()
        self.m1.set_direction(True)
        self.m2.set_direction(True)
        self.direction = True

    def light_on(self):
        self.led.on()

    def light_off(self):
        self.led.off()

    def change_direction(self, d=True):
        self.direction = d
        self.step_counter = self.tension_steps
        self.m1.set_direction(d)
        self.m2.set_direction(d)

    def steps_forward(self, steps=1):
        if not self.direction:
            self.change_direction(True)
        original_steps = steps
        m1step = self.m1.step
        m2step = self.m2.step
        while steps > 1:
            steps -= 1
            if self.step_counter > 0:
                m1step()
                self.step_counter -= 1
            else:
                self.step_counter = self.tension_steps
            m2step()
        feed_pulses, takeup_pulses = self._schedule_reel_pulses(original_steps)
        self._run_deferred_reel_pulses(feed_pulses, takeup_pulses, original_steps)

    def steps_back(self, steps=1):
        if self.direction:
            self.change_direction(False)
        original_steps = steps
        m1step = self.m1.step
        m2step = self.m2.step
        while steps > 1:
            steps -= 1
            if self.step_counter > 0:
                m2step()
                self.step_counter -= 1
            else:
                self.step_counter = self.tension_steps
            m1step()
        feed_pulses, takeup_pulses = self._schedule_reel_pulses(original_steps)
        self._run_deferred_reel_pulses(feed_pulses, takeup_pulses, original_steps)

    def _schedule_reel_pulses(self, advance_steps):
        self.feed_counter += advance_steps
        feed_pulses = self.feed_counter // self.feed_interval
        self.feed_counter = self.feed_counter % self.feed_interval

        self.take_up_counter += advance_steps
        takeup_pulses = self.take_up_counter // self.take_up_steps
        self.take_up_counter = self.take_up_counter % self.take_up_steps

        return int(feed_pulses), int(takeup_pulses)

    def _run_deferred_reel_pulses(self, feed_pulses, takeup_pulses, advance_steps):
        print(f"[APP] Advance complete: steps={advance_steps}")
        time.sleep(self.advance_settle_delay)
        if feed_pulses <= 0 and takeup_pulses <= 0:
            return

        if feed_pulses > 0:
            print(f"[APP] Running feed reel pulses: count={feed_pulses}, steps={feed_pulses * self.feed_interval}")
            for _ in range(feed_pulses):
                self.reel1.pulse(self.feed_pulse_delay)

        if takeup_pulses > 0:
            print(f"[APP] Running take-up reel pulses: count={takeup_pulses}, steps={takeup_pulses * self.take_up_steps}")
            for _ in range(takeup_pulses):
                self.reel2.pulse(self.takeup_pulse_delay)

        print("[APP] Deferred reel pulses complete")
        time.sleep(self.post_takeup_settle_delay)

    def tension_film(self, steps=200):
        d = self.direction
        self.m1.set_direction(False)
        self.m2.set_direction(True)
        for _ in range(steps):
            self.m1.step()
            self.m2.step()
        self.m1.set_direction(d)
        self.m2.set_direction(d)

    def clean_up(self):
        self.led.off()
        self.reel1.off()
        self.reel2.off()
        self.m1.off()
        self.m2.off()
        lgpio.spi_close(self.h_spi)

class stepperMotor:
    delay = 0.001
    rotation_steps = 3200

    def __init__(self, step_pin, dir_pin, en_pin, h_spi, addr):
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.en_pin = en_pin
        self.h_spi = h_spi
        self.addr = addr
        self.on()
        self.direction = True

    def on(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, 0]))
        self.motor_on = True

    def off(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, 1 << (self.en_pin - tcControl.pin_base)]))
        self.motor_on = False

    def set_direction(self, direction=True):
        self.direction = direction
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, (1 << (self.dir_pin - tcControl.pin_base)) if not direction else 0]))

    def step(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, 1 << (self.step_pin - tcControl.pin_base)]))
        time.sleep(self.delay)
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, 0]))

    def steps(self, s):
        for _ in range(s):
            self.step()

    def rotate_full(self):
        self.steps(self.rotation_steps)

    def rotate_half(self):
        self.steps(self.rotation_steps // 2)

    def rotate_quarter(self):
        self.steps(self.rotation_steps // 4)

class ledControl:
    def __init__(self, pin, h_spi, addr):
        self.pin = pin
        self.h_spi = h_spi
        self.addr = addr
        self.off()

    def on(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, 1 << (self.pin - tcControl.pin_base)]))

    def off(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, 0]))

class reelMotor:
    def __init__(self, pin, h_spi, addr):
        self.pin = pin
        self.h_spi = h_spi
        self.addr = addr
        self.off()

    def on(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, 1 << (self.pin - tcControl.pin_base)]))

    def off(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOA, 0]))

    def pulse(self, duration):
        self.on()
        time.sleep(duration)
        self.off()

class shutterRelease:
    wake_delay = 1.5
    shutter_delay = 0.3

    def __init__(self, focus_pin, shutter_pin, h_spi, addr):
        self.focus_pin = focus_pin
        self.shutter_pin = shutter_pin
        self.h_spi = h_spi
        self.addr = addr
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOB, 0]))

    def wake_camera(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOB, 1 << (self.focus_pin - tcControl.pin_base)]))
        time.sleep(self.wake_delay)
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOB, 0]))

    def fire_shutter(self):
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOB, 1 << (self.shutter_pin - tcControl.pin_base)]))
        time.sleep(self.shutter_delay)
        lgpio.spi_write(self.h_spi, bytes([0x40 | (self.addr << 1), tcControl.MCP23S17_GPIOB, 0]))
