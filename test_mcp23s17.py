import unittest

import mcp23s17


class FakeLgpio:
    def __init__(self):
        self.opens = []
        self.writes = []
        self.closes = []

    def spi_open(self, *args):
        self.opens.append(args)
        return 7

    def spi_write(self, handle, data):
        self.writes.append((handle, bytes(data)))
        return len(data)

    def spi_close(self, handle):
        self.closes.append(handle)
        return 0


class Mcp23s17Tests(unittest.TestCase):
    def make_device(self):
        fake = FakeLgpio()
        return mcp23s17.MCP23S17(spi_module=fake), fake

    def test_spi_open_and_register_configuration(self):
        device, fake = self.make_device()
        self.assertEqual(fake.opens, [(0, 0, 1_000_000, 0)])
        self.assertEqual(
            fake.writes[:4],
            [
                (7, bytes((0x40, 0x12, 0))),
                (7, bytes((0x40, 0x13, 0))),
                (7, bytes((0x40, 0x00, 0xFF))),
                (7, bytes((0x40, 0x01, 0xFF))),
            ],
        )
        device.configure_output(0)
        device.configure_output(8)
        self.assertEqual(fake.writes[-2:], [
            (7, bytes((0x40, 0x00, 0xFE))),
            (7, bytes((0x40, 0x01, 0xFE))),
        ])

    def test_spi_opcode_maps_configured_mcp_address(self):
        self.assertEqual(mcp23s17.MCP23S17._write_opcode(0x20), 0x40)
        self.assertEqual(mcp23s17.MCP23S17._write_opcode(0x23), 0x46)
        with self.assertRaises(ValueError):
            mcp23s17.MCP23S17._write_opcode(0x28)

    def test_native_pin_mapping_selects_port_and_bit(self):
        self.assertEqual(mcp23s17.MCP23S17._pin_location(0), ("a", 1))
        self.assertEqual(mcp23s17.MCP23S17._pin_location(7), ("a", 0x80))
        self.assertEqual(mcp23s17.MCP23S17._pin_location(8), ("b", 1))
        self.assertEqual(mcp23s17.MCP23S17._pin_location(15), ("b", 0x80))
        with self.assertRaises(ValueError):
            mcp23s17.MCP23S17._pin_location(16)

    def test_gpio_a_writes_preserve_unrelated_bits(self):
        device, fake = self.make_device()
        device.write(0, 1)
        device.write(7, 1)
        device.write(7, 0)
        self.assertEqual(device.gpio_a, 1)
        self.assertEqual(fake.writes[-1], (7, bytes((0x40, 0x12, 1))))

    def test_gpio_b_writes_preserve_unrelated_bits(self):
        device, fake = self.make_device()
        device.write(8, 1)
        device.write(10, 1)
        device.write(8, 0)
        self.assertEqual(device.gpio_b, 1 << 2)
        self.assertEqual(fake.writes[-1], (7, bytes((0x40, 0x13, 1 << 2))))

    def test_spi_close_is_idempotent(self):
        device, fake = self.make_device()
        device.close()
        device.close()
        self.assertEqual(fake.closes, [7])

    def test_missing_lgpio_fails_only_when_device_is_constructed(self):
        original = mcp23s17.lgpio
        try:
            mcp23s17.lgpio = None
            with self.assertRaises(RuntimeError):
                mcp23s17.MCP23S17()
        finally:
            mcp23s17.lgpio = original


if __name__ == '__main__':
    unittest.main()
