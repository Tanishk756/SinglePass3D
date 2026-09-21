# Telemetry format

CSV header or JSON object fields: `timestamp,latitude,longitude,altitude`. Optional fields: `roll,pitch,yaw,velocity_x,velocity_y,velocity_z`. Timestamp is timezone-aware ISO-8601 or Unix seconds. Latitude and longitude are WGS84 degrees. Altitude is numeric meters in the source datum; document whether it is ellipsoidal or orthometric.

The video start time must refer to frame zero in the same UTC clock. Nearest and linear interpolation are supported, with a tolerance. Missing matches remain explicit in `frame_telemetry.csv`. Telemetry samples must have unique timestamps. Format adapters for DJI, ArduPilot, and PX4 are future work.
