# Raspberry Pi 4 64-bit migration inventory

This is the read-only inventory of `pi4` used as the reference for moving the
telecine controller to a fresh current 64-bit Raspberry Pi OS SD card.

- Inventory date: 2026-09-21 (America/Detroit)
- Source host: `pi4` / `pi4.lan`; hostname `pi4`
- Existing installation: Raspbian/Raspberry Pi OS Bullseye, Debian 11
- Mac repository: `/Users/todd/bin/GrokCam`
- Pi repository: `/home/williatf/bin/GrokCam`

The document is at `docs/PI4_64BIT_MIGRATION_INVENTORY.md`. The repository had
no established top-level documentation directory (only experiment READMEs),
so `docs/` is the appropriate dedicated location for a durable migration and
operations reference.

Inspection was read-only. No package, file, service, GPIO, transport, camera,
or repository state was changed on `pi4`. Camera-list and capture commands were
not run because the production service was active and the requested safety rule
prohibited interfering with camera access. Camera facts below use the explicit
boot overlay, running application source, installed package metadata, device
nodes, and existing capture records.

## Executive migration requirements

The new card must preserve:

1. A Pi 4 with the IMX477 connected and a working camera overlay. The
   production path is 2028x1520, 12-bit Bayer RAW, DNG output.
2. The `williatf` runtime account, its device groups, and access to SPI, the
   MCP23S17 controller, camera/media devices, and `/mnt/SG1TB`.
3. The SG1TB disk mounted before GrokCam starts. GrokCam fails at import time
   when this mount is absent.
4. Ignored machine state: `config.json`, `crop_config.json`,
   `calibration.json`, `calibration.super8.json`, backups, and per-project
   metadata/camera settings.
5. The systemd unit and its system-Python interpreter, working directory, and
   restart policy.
6. The camera and transport validation gates in this document. A web page
   loading is not proof that RAW DNG capture or transport control works.

## 1. Hardware and architecture

| Item | Observed value |
| --- | --- |
| Board | Raspberry Pi 4 Model B Rev 1.1 |
| Device-tree revision | `c03111` |
| CPU | BCM2711, four ARMv8 cores |
| Kernel | `6.1.21-v8+ #1642 SMP PREEMPT`, `aarch64` |
| Userland word size | 32-bit (`getconf LONG_BIT` = 32) |
| Debian architecture | `armhf` |
| RAM | 3.7 GiB visible to Linux |
| Hostname | `pi4` |
| Camera | IMX477 overlay configured |
| External disk | Seagate USB portable disk, 1 TB class, `/dev/sda1`, label `SG1TB` |
| USB | Seagate disk, Genesys/VIA hubs, Logitech Unifying receiver |
| GPIO | `/dev/gpiochip0`, `/dev/gpiochip1` |
| SPI | `/dev/spidev0.0`, `/dev/spidev0.1` |
| I2C | `/dev/i2c-0`, `/dev/i2c-1`, `/dev/i2c-10`, `/dev/i2c-22` |
| Camera/media | `/dev/media0` through `/dev/media4` and many `/dev/video*` |
| UART | `/dev/ttyAMA0` |

The unusual existing combination is confirmed: this Pi is 64-bit capable and
boots a 64-bit kernel, but its Debian/Raspbian userland and dpkg packages are
`armhf`. The target should use a native `arm64` userland; all camera, GPIO, and
Python extensions must be revalidated rather than copied as armhf binaries.

The external USB disk is the only storage device relevant to capture. Keep the
USB hub chain and disk connected during storage and sustained-write validation.

## 2. Operating system, firmware, and boot

The installed OS reports:

~~~text
PRETTY_NAME="Raspbian GNU/Linux 11 (bullseye)"
VERSION_ID="11"
VERSION_CODENAME=bullseye
/etc/debian_version: 11.11
~~~

The kernel command line is:

~~~text
console=serial0,115200 console=tty1 root=PARTUUID=d280256d-02 rootfstype=ext4 fsck.repair=yes rootwait quiet splash plymouth.ignore-serial-consoles
~~~

The existing boot partition is `/boot` (VFAT), not the newer
`/boot/firmware` layout. Relevant `/boot/config.txt` settings are:

~~~ini
dtparam=i2c_arm=on
dtparam=spi=on
dtparam=audio=on
start_x=1
max_framebuffers=2
disable_overscan=1

[pi4]
camera_auto_detect=1
dtoverlay=vc4-fkms-v3d
dtoverlay=imx477
arm_boost=1

[all]
gpu_mem=128
~~~

The new OS may put the equivalent file under `/boot/firmware/config.txt`; do
not assume the old path. Preserve functional settings, but let the current
Raspberry Pi OS camera stack determine whether the explicit `imx477` overlay is
still required.

Firmware and boot packages:

- `vcgencmd version`: 2023-03-17 firmware, hash
  `82f3750a65fadae9a38077e3c2e217ad158c8d54` (clean, `start_x`)
- `raspberrypi-kernel`: `1:1.20230405-1`
- `raspberrypi-bootloader`: `1:1.20230405-1`
- `rpi-eeprom`: `16.1-1`
- `vcgencmd get_throttled` observed `0x50005`; keep this as a baseline and
  investigate power/undervoltage history if it recurs.

## 3. Camera stack and IMX477 RAW behavior

Installed Raspberry Pi camera packages:

| Package | Installed version | Source |
| --- | --- | --- |
| `libcamera0:armhf` | `0~git20230720+bde9b04f-1` | apt |
| `python3-libcamera` | `0~git20230720+bde9b04f-1` | apt |
| `libcamera-apps` | `1.2.1-1` | apt |
| `python3-picamera2` | `0.3.12-2` | apt |

Picamera2 is imported from
`/usr/lib/python3/dist-packages/picamera2` and libcamera's Python extension is
the `arm-linux-gnueabihf` build under the same dist-packages tree. The running
service does not use a virtualenv. The pinned requirement
`picamera2==0.3.12` is an inventory record, not a safe install recipe for a
current 64-bit OS. Prefer the current OS apt-provided Picamera2/libcamera pair
and record the versions tested; do not mix a pip Picamera2 with another
libcamera ABI.

Production constants are in [app.py](../app.py:48):

~~~python
RAW_CAPTURE_MODE = 'raw_dng_v1'
RAW_SENSOR_SIZE = (2028, 1520)
RAW_PREVIEW_SIZE = (760, 570)

camera.create_still_configuration(
    main={"size": RAW_PREVIEW_SIZE, "format": "RGB888"},
    raw={"size": RAW_SENSOR_SIZE, "format": "SRGGB12"},
    buffer_count=2,
)
~~~

This behavior must be preserved:

- sensor RAW: 2028x1520, `SRGGB12` (12-bit Bayer)
- preview: 760x570, `RGB888`
- two buffers for the RAW configuration
- Picamera2 request capture followed by `request.save_dng(...)`
- output: `<project>/raw/frame_######.dng`
- write `.partial.dng`, then atomically rename to the final `.dng`
- drain old requests and check `SensorTimestamp` after transport settling

The legacy/focus/Regular 8 path configures a still main stream at the active
calibration resolution and JPEG quality 90; it is not the archival RAW path.
The DNG implementation is in [app.py](../app.py:1617) and
[app.py](../app.py:1667).

### Camera controls

Production manual controls in [app.py](../app.py:776) are:

- `ExposureTime`, clamped to 100–50,000 microseconds
- `AnalogueGain`, clamped to 1.0–16.0
- `AeEnable=False` and `AwbEnable=False`
- optional persisted `ColourGains`

One-shot AE can be enabled, metadata read, and the result locked back to
manual exposure/gain. One-shot AWB can similarly read `ColourGains` and lock
the result. Focus preview temporarily enables AE with AWB off. Saved project
settings are preferred for capture. When a project has no saved camera settings,
RAW capture uses the shared camera-setting default of 3300 microseconds at gain
1.0, so preview and DNG capture do not silently diverge.

Current machine calibration on the Pi:

~~~json
{
  "calibration_version": 2,
  "calibration_resolution": [2028, 1520],
  "exposure_time": 1274,
  "gain": 1.0,
  "sprocket_pitch_px": 781.5,
  "steps_per_pitch": 277,
  "steps_per_px": 0.3544465770953295,
  "sprocket_area_min": 87848,
  "sprocket_area_max": 97095
}
~~~

Super 8 is separate: exposure 878 microseconds, gain 1.0, 308 steps per pitch,
`sprocket_pitch_px` 853.3922995942081, and detector limits 31,320–39,861
pixels. The full quality measurements remain in
`/home/williatf/bin/GrokCam/calibration.super8.json`.

## 4. GrokCam repository and source configuration

The Pi service runs `/home/williatf/bin/GrokCam`.

- branch: `main`, tracking `origin/main`
- HEAD: `9c3dbe0a7a96f601912b46504f347fb7aea120fe`
- commit: `Implemented bidirectional low-disturbance adaptation. (2026-09-16)`
- remote: `git@github.com:williatf/GrokCam.git`
- no tracked modifications
- untracked: `requirements.pi4.backup.txt`
- ignored: runtime config/calibration files, backups, and `__pycache__/`
- no tags in the inspected clone

The Mac checkout was clean on `main` before this document was added. The
repository deliberately ignores machine-specific production state, so a clone
alone is insufficient.

The Mac-side ignored runtime files are not assumed to be identical to the Pi's
production files. Use the values and files inventoried from `pi4` as the
authoritative migration source, and compare checksums/content before copying.

| File | Purpose |
| --- | --- |
| [app.py](../app.py:48) | Picamera2, controls, RAW DNG, metadata, projects, WebSocket app |
| [control.py](../control.py:1) | **Active** transport: wiringpi + MCP23S17 |
| [control_lgpio.py](../control_lgpio.py:1) | Alternative lgpio/SPI transport; not imported by active app |
| [registration.py](../registration.py:1) | Registration tracking |
| [sprocket.py](../sprocket.py:1), [fast_sprocket.py](../fast_sprocket.py:1) | Regular 8 detection |
| [super8_detector.py](../super8_detector.py:1), [super8_phase_tracker.py](../super8_phase_tracker.py:1) | Super 8 detection/phase |
| [film_calibration.py](../film_calibration.py:1) | Film calibration and preview scaling |
| `config.json`, `crop_config.json` | Ignored crop state |
| `calibration.json`, `calibration.super8.json` | Ignored machine calibration |
| project `metadata.json` | Project format, crop, transport state |
| project `camera_settings.json` | Optional exposure/gain/colour-gain state |

The current Pi `config.json` contains crop geometry relative to registration:
`x1=308`, `x2=1694`, `y_offset=-591.25`, `height=984`,
`registration_y=860.25`, source resolution `2028x1520`. The normalized crop file
is:

~~~json
{"crop_norm":[0.12721893491124261,0.1730263157894737,0.4383629191321499,0.7023026315789473]}
~~~

## 5. Python environment and dependency provenance

The active interpreter is `/usr/bin/python3` / Python 3.9.2. The service calls
it directly. A user virtual environment exists at `/home/williatf/.venv`,
also Python 3.9.2, with `include-system-site-packages = false`; its only
relevant installed package observed was `numpy==2.0.2`. It is not used by
GrokCam.

Relevant system-environment versions:

| Component | Version | Use/provenance |
| --- | --- | --- |
| Python | 3.9.2 | service interpreter |
| Picamera2/libcamera | 0.3.12 / 20230720 snapshot | apt |
| NumPy | 1.19.5 | system environment |
| OpenCV | 4.11.0.86 | pip-visible |
| Pillow | 8.1.2 | pip-visible |
| pidng | 4.0.9 | DNG support |
| Flask / Flask-SocketIO | 1.1.2 / 5.0.1 | web app |
| websockets | 15.0.1 | asyncio WebSockets |
| eventlet / gevent | 0.26.1 / 25.9.1 | installed app dependencies |
| wiringpi | 2.60.1 | required by active `control.py` |
| lgpio | 0.2.2.0 | alternative `control_lgpio.py` |
| RPi.GPIO | 0.7.0 | installed, not active |
| spidev | 3.5 | installed |
| pyserial | 3.5b0 | installed; no active import found |

The repository `requirements.txt` and untracked Pi-side
`requirements.pi4.backup.txt` are broad old Raspberry Pi freezes. Several pins
predate current Python, NumPy, Pillow, and libcamera packages. On the new
card, install current OS camera/GPIO packages first, add application
dependencies deliberately, and record the final apt package list and
`pip freeze`.

## 6. Telecine hardware interfaces

The active app imports `tcControl` from `control.py`. It calls legacy wiringpi
and configures an MCP23S17 at address `0x20` on SPI channel 0 / CE0:
`wiringpi.mcp23s17Setup(100, 0, 0x20)`.

| Expander logical pin | Function |
| ---: | --- |
| 100 | pusher stepper enable |
| 101 | pusher step |
| 102 | pusher direction |
| 103 | puller direction |
| 104 | puller step |
| 105 | puller stepper enable |
| 106 | feed reel |
| 107 | take-up reel |
| 108 | illumination LED |
| 109 | declared shutter/focus output |
| 110 | declared shutter/focus output |

The active implementation drives both steppers, feed/take-up reels, and LED.
It uses pusher ratio 0.98, feed pulses every 5000 transport steps, take-up
pulses every 2500 steps normally, and capture-local adaptive take-up timing in
RAW mode. `control_lgpio.py` is a complete alternative using `lgpio.spi_open(0)`
and the same expander scheme, but it is not imported by the running app.
Changing to it is a code change requiring a full transport test.

Boot enables physical I2C and SPI with `dtparam=i2c_arm=on` and
`dtparam=spi=on`. No active GrokCam import of I2C, UART, PWM, or a standalone
GPIO sensor was found; exposed device nodes alone do not establish an app
dependency.

## 7. Services, startup, and permissions

`grokcam.service` is enabled and running:

~~~ini
[Unit]
Description=GrokCam Flask App
After=network-online.target
Wants=network-online.target

[Service]
User=williatf
WorkingDirectory=/home/williatf/bin/GrokCam
ExecStart=/usr/bin/python3 /home/williatf/bin/GrokCam/app.py
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
~~~

The observed process was `/usr/bin/python3 /home/williatf/bin/GrokCam/app.py`,
listening on TCP port 5000. It was not stopped or restarted during inspection.

Other enabled/running baseline services include `ssh`, `dhcpcd`, `cron`,
`udev`, `wpa_supplicant`, `smbd`, `nmbd`, `nfs-server`, `netatalk`,
`udisks2`, and `lightdm`. No user crontab, GrokCam rc.local launch, user
autostart entry, or custom GrokCam service was found beyond `grokcam.service`.
`/etc/rc.local` only prints the IP. tty1 autologin is configured for
`williatf`, independently of GrokCam.

The runtime identity is:

~~~text
uid=1000(williatf) gid=1000(williatf)
groups: adm,dialout,cdrom,sudo,audio,video,plugdev,games,users,input,render,netdev,spi,i2c,gpio,lpadmin
~~~

Relevant device permissions are group based: camera/media nodes
`root:video` 0660; SPI `root:spi` 0660; I2C `root:i2c` 0660; GPIO
`root:gpio` 0660; UART `root:dialout` 0660. The custom
`/etc/udev/rules.d/99-com.rules` assigns these groups and grants GPIO/PWM
sysfs access to `gpio`. Recreate the equivalent current rules and group
membership before starting the service.

## 8. Storage and capture data

| Device | Filesystem | UUID/label | Mount |
| --- | --- | --- | --- |
| `/dev/mmcblk0p1` | VFAT | `C336-AC83` / `bootfs` | `/boot` |
| `/dev/mmcblk0p2` | ext4 | `eaaa4faa-eab6-400c-950f-dc96ae4e0400` / `rootfs` | `/` |
| `/dev/sda1` | ext4 | `e172dcce-267d-44de-87f0-2110c41404ec` / `SG1TB` | `/mnt/SG1TB` |

The SG1TB disk is approximately 916 GiB usable, 426 GiB used, 444 GiB free
at inventory time. The root filesystem is approximately 29 GiB with 9.5 GiB
used. Its fstab entry is:

~~~fstab
UUID=e172dcce-267d-44de-87f0-2110c41404ec /mnt/SG1TB ext4 defaults,nofail,x-systemd.device-timeout=10 0 2
~~~

GrokCam writes:

~~~text
/mnt/SG1TB/GrokCam/projects/<project>/
    metadata.json
    camera_settings.json
    frames/
    raw/frame_######.dng
    debug/*.jsonl, *.jpg
~~~

The disk contains the `GrokCam` tree (about 308 GiB), 20 project directories,
test frames, RAW DNG projects, legacy frames, videos, manifests, calibration
metadata, and many registration/RAW JSONL records. Preserve the complete tree,
not only the active project. The active project at inventory time was
`Unknown_Super8_2`, `film_format: super8`, with calibrated transport state.

Mount ownership is `williatf:williatf`; `/mnt/SG1TB` is mode 777 and
GrokCam/project directories are mode 755. Samba share `[SG1TB]` and AFP share
`AFP SG1TB` expose the disk. NFS exports `/mnt/SG1TB` read/write to
`192.168.4.0/22`. Recreate only the shares actually needed; do not copy
credentials or private keys into this document.

## 9. Networking and administration

Observed interfaces and DHCP addresses:

- Ethernet `eth0`: `192.168.4.52/22`
- Wi-Fi `wlan0`: `192.168.6.126/22`
- gateway: `192.168.4.1`
- IPv6 addresses are also assigned

`/etc/dhcpcd.conf` has default DHCP settings and no static address stanza.
Hostname is `pi4`; mDNS advertises `pi4.local`. SSH is enabled on port 22.
The inventory intentionally does not record passwords, Wi-Fi PSKs, authorized
keys, host keys, or other secrets.

GrokCam listens on port 5000. SMB, NFS, AFP, and SSH are administration/data
services. Retain a predictable hostname and either a DHCP reservation or an
equivalent name/address workflow so `ssh pi4` and the browser client remain
usable.

## 10. Calibration and machine state to preserve

Copy and verify:

1. `config.json`
2. `crop_config.json`
3. `calibration.json`
4. `calibration.super8.json`
5. Historical calibration backups if recovery matters
6. Every project `metadata.json` and `camera_settings.json`
7. RAW/registration JSONL records and manifests used to audit captures
8. Pi-side `requirements.pi4.backup.txt`

Regular 8: 2028x1520 geometry, exposure 1274, gain 1.0, 781.5 pixels per
pitch, 277 steps per pitch. Super 8: exposure 878, gain 1.0, 853.3923 pixels
per pitch, 308 steps per pitch. These are physical-machine state, not generic
defaults.

## 11. Fresh-card build checklist

1. Install current 64-bit Raspberry Pi OS. Confirm `uname -m` is `aarch64`,
   `getconf LONG_BIT` is 64, and dpkg architecture is `arm64`.
2. Set the hostname and SSH/DHCP workflow used by the Mac.
3. Enable SPI and I2C. Configure the IMX477 through the current OS camera
   configuration, preserving the explicit overlay only if required.
4. Install the current apt-compatible Picamera2/libcamera pair; do not mix
   architectures or camera ABI versions.
5. Recreate `williatf` with the UID/GID and `video`, `spi`, `i2c`, `gpio`,
   `dialout`, `input`, `render`, and storage memberships above.
6. Recreate equivalent udev group rules for camera/media, SPI, I2C, GPIO, PWM,
   and UART devices.
7. Mount SG1TB by UUID at `/mnt/SG1TB` and copy/verify the complete data tree.
8. Clone at commit `9c3dbe0a7a96f601912b46504f347fb7aea120fe` (or an
   intentionally selected later commit), then restore ignored runtime files.
9. Decide explicitly whether to retain/test `wiringpi` on arm64 or migrate to
   `control_lgpio.py`; do not change this implicitly.
10. Recreate the systemd unit and require the SG1TB mount before GrokCam starts.

## 12. Validation gates

1. **Architecture:** 64-bit kernel, `arm64` userland, arm64-compatible camera
   and GPIO extension paths.
2. **Devices:** camera/media, SPI, I2C, GPIO, and UART nodes exist with the
   expected group permissions.
3. **Camera discovery:** on the idle new system, confirm one IMX477 and the
   2028x1520 RAW mode before starting GrokCam.
4. **Picamera2 smoke test:** create the two-stream production configuration,
   apply manual controls, acquire one request, inspect metadata, and save one
   temporary DNG. Confirm Bayer dimensions, bit depth, and DNG readability.
5. **Storage:** test read/write on SG1TB, free space, ownership, sustained
   writes, and mount ordering before service startup.
6. **Transport:** with no film loaded, verify both stepper directions, pusher/
   puller relation, LED, feed reel, take-up reel, and cleanup output state.
7. **Application:** confirm restored calibration/project state loads and RAW
   capture does not silently use `raw_safe_default`.
8. **RAW capture:** short Regular 8 and Super 8 captures must create
   2028x1520 DNGs, 760x570 previews, camera metadata JSONL, and no leftover
   `.partial.dng` files.
9. **Calibration:** compare pitch, registration error, commanded steps, crop
   geometry, and Super 8 phase/crop metadata with preserved values.
10. **Startup/recovery:** after the checks pass, reboot and verify the mount is
    present before `grokcam.service`, port 5000 listens, and camera/GPIO
    cleanup survives a service restart.

Keep the original SD card and original SG1TB data untouched until the new card
passes this sequence and a known-good sample has been reviewed.
