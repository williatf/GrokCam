"""Small MCP23S17 GPIO expander driver for the GrokCam transport."""

try:
    import lgpio
except ImportError:  # pragma: no cover - exercised by the macOS test setup
    lgpio = None


class MCP23S17:
    """Stateful MCP23S17 access over SPI using native pin numbers 0-15."""

    SPI_DEVICE = 0
    SPI_CHANNEL = 0
    SPI_BAUD = 1_000_000
    SPI_FLAGS = 0  # mode 0, active-low CE0

    IODIRA = 0x00
    IODIRB = 0x01
    GPIOA = 0x12
    GPIOB = 0x13

    def __init__(self, address=0x20, spi_module=None):
        self.address = int(address)
        self.write_opcode = self._write_opcode(self.address)
        self._lgpio = spi_module if spi_module is not None else lgpio
        if self._lgpio is None:
            raise RuntimeError("lgpio is required to initialize the MCP23S17")

        self._handle = self._lgpio.spi_open(
            self.SPI_DEVICE,
            self.SPI_CHANNEL,
            self.SPI_BAUD,
            self.SPI_FLAGS,
        )
        self._closed = False
        self.gpio_a = 0
        self.gpio_b = 0
        self.iodir_a = 0xFF
        self.iodir_b = 0xFF

        # Initialize latches low while the expander pins are still inputs.
        self._write_register(self.GPIOA, self.gpio_a)
        self._write_register(self.GPIOB, self.gpio_b)
        self._write_register(self.IODIRA, self.iodir_a)
        self._write_register(self.IODIRB, self.iodir_b)

    @staticmethod
    def _write_opcode(address):
        """Convert the configured 0x20-0x27 address to an SPI write opcode."""
        address = int(address)
        if not 0x20 <= address <= 0x27:
            raise ValueError("MCP23S17 address must be in the range 0x20-0x27")
        hardware_address_bits = address - 0x20
        return 0x40 | (hardware_address_bits << 1)

    @staticmethod
    def _pin_location(pin):
        pin = int(pin)
        if not 0 <= pin <= 15:
            raise ValueError("MCP23S17 pin must be in the range 0-15")
        if pin < 8:
            return "a", 1 << pin
        return "b", 1 << (pin - 8)

    def _write_register(self, register, value):
        if self._closed:
            raise RuntimeError("MCP23S17 SPI device is closed")
        self._lgpio.spi_write(
            self._handle,
            bytes((self.write_opcode, int(register) & 0xFF, int(value) & 0xFF)),
        )

    def configure_output(self, pin):
        port, bit = self._pin_location(pin)
        if port == "a":
            self.iodir_a &= ~bit
            self._write_register(self.IODIRA, self.iodir_a)
        else:
            self.iodir_b &= ~bit
            self._write_register(self.IODIRB, self.iodir_b)

    def write(self, pin, value):
        port, bit = self._pin_location(pin)
        if port == "a":
            self.gpio_a = (self.gpio_a | bit) if value else (self.gpio_a & ~bit)
            self._write_register(self.GPIOA, self.gpio_a)
        else:
            self.gpio_b = (self.gpio_b | bit) if value else (self.gpio_b & ~bit)
            self._write_register(self.GPIOB, self.gpio_b)

    def close(self):
        if not self._closed:
            self._lgpio.spi_close(self._handle)
            self._closed = True
