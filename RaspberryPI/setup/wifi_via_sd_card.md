# Wi-Fi setup via SD card (no monitor / keyboard needed)

Use this when you have the Pi but no HDMI display or keyboard at the new
location, and you need to give it Wi-Fi credentials before first boot.

## Requirements
- The Pi powered off
- An SD card reader on your laptop
- The Wi-Fi SSID and password for the new location

## Steps

1. **Power off the Pi**, eject the microSD, plug it into your laptop. The
   `boot` partition (FAT, ~256 MB) will mount automatically on Windows / macOS.

2. **Create `wpa_supplicant.conf`** in the root of the `boot` partition with
   exactly this content (replace the SSID and password):

   ```conf
   country=US
   ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
   update_config=1

   network={
       ssid="YOUR_SSID"
       psk="YOUR_PASSWORD"
       key_mgmt=WPA-PSK
   }
   ```

   On first boot, Raspberry Pi OS moves this file into
   `/etc/wpa_supplicant/wpa_supplicant.conf` automatically. After the move,
   the Wi-Fi connects within ~20 seconds.

3. **Create an empty file named `ssh`** (no extension) in the same `boot`
   partition. This re-enables the SSH server, which is disabled by default
   on fresh Pi OS images.

4. **Safely eject** the SD card and put it back in the Pi.

5. **Power on the Pi**. Wait ~60 seconds for it to boot, join Wi-Fi, and
   start `lenet1.service`.

## Finding the Pi on the new network

The IP will be assigned by the new router (the previous `192.168.45.76`
was a PC-shared ICS address that no longer applies).

- **Easiest:** open the router admin page → look for the device named
  `lenet`. That's its hostname.
- **mDNS:** `ssh lenet@lenet.local` works if avahi is enabled on both sides.
- **Network scan:** `arp -a` on the same LAN, or
  `nmap -sn 192.168.1.0/24` (adjust subnet). Pi MAC prefixes:
  `b8:27:eb`, `dc:a6:32`, `d8:3a:dd`, `e4:5f:01`.

## Connecting

```bash
ssh -i path/to/lenet_id_ed25519 lenet@<pi-ip>
# or with password
ssh lenet@<pi-ip>     # password: nikonalex
```

The twin web UI is at `http://<pi-ip>:8080/` once `lenet1.service` is up
(systemd starts it automatically on boot).

## Multiple networks (optional)

To pre-load several known networks (home + new location + backup hotspot),
add more `network={ ... }` blocks to `wpa_supplicant.conf`. The Pi picks
the strongest one it can see.

## Troubleshooting

- **No IP after boot:** the SSID/password is wrong, or the Pi is out of
  range. SSH won't be reachable; you'll need a monitor + keyboard to
  diagnose (`sudo nmcli device wifi list`, `journalctl -u wpa_supplicant`).
- **`wpa_supplicant.conf` ignored:** make sure it's at the root of the
  `boot` partition (not inside a subfolder) and uses LF line endings.
- **`ssh` file ignored:** same — must be at the root of `boot`, no
  extension.
- **Service not running on port 8080:** `sudo systemctl status lenet1` to
  check. If it's failed, `journalctl -u lenet1 --no-pager | tail -40`.
