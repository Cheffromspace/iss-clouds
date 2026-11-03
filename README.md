# ISS world live clouds

Hourly global cloud images for the ISS VRChat world, made from NOAA's Global Mosaic of Geostationary Satellite Imagery (GMGSI, longwave infrared), which is public. The workflow converts the latest hour with `live_clouds.py` and publishes it to GitHub Pages:

- https://cheffromspace.github.io/iss-clouds/west.png (180° W to 0°)
- https://cheffromspace.github.io/iss-clouds/east.png (0° to 180° E)
- https://cheffromspace.github.io/iss-clouds/time.txt (the hour shown, UTC)

Each image is 2048 × 2048 RGB, equirectangular from 90° N to 90° S. Red is cloud opacity; green is cloud-top height from 0 to 16 km; blue is a constant 128 (a marker for detecting sRGB decoding). Data: NOAA/NESDIS.
