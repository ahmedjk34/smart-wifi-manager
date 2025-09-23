# ================== SERVER COMMANDS ==================

# 1. Create configuration file

python3 python_files/6a_ml_inference_server.py --create-config

# 2. Start enhanced server (basic)

python3 python_files/6a_ml_inference_server.py --config server_config.json

# 3. Start server with custom port and debug logging

python3 python_files/6a_ml_inference_server.py --config server_config.json --port 9000 --log-level DEBUG

# 4. Background server with logging

python3 python_files/6a_ml_inference_server.py --config server_config.json > server.log 2>&1 &

# ================== CLIENT COMMANDS ==================

# 5. Get server info

python3 python_files/6b_ml_client.py --info

# 6. List available models

python3 python_files/6b_ml_client.py --models

# 7. Test server connectivity

python3 python_files/6b_ml_client.py --ping

# 8. Get server statistics

python3 python_files/6b_ml_client.py --stats --format json

# 9. Single prediction with oracle_balanced model (28 safe features)

python3 python_files/6b_ml_client.py --features 25.0 24.0 23.0 1.2 2.5 0.8 0.75 0.70 5 2 0.1 0.85 3 1500 0.95 0.3 0.8 100 200 300 1 1 1000000 50 3 20 0.5 2.5 --model oracle_balanced

# 10. Prediction with compact output format

python3 python_files/6b_ml_client.py --features 25.0 24.0 23.0 1.2 2.5 0.8 0.75 0.70 5 2 0.1 0.85 3 1500 0.95 0.3 0.8 100 200 300 1 1 1000000 50 3 20 0.5 2.5 --format compact

# 11. Load features from file

echo "25.0 24.0 23.0 1.2 2.5 0.8 0.75 0.70 5 2 0.1 0.85 3 1500 0.95 0.3 0.8 100 200 300 1 1 1000000 50 3 20 0.5 2.5" > features.txt
python3 python_files/6b_ml_client.py --features-file features.txt --format detailed

# 12. Benchmark performance (100 predictions)

python3 python_files/6b_ml_client.py --features 25.0 24.0 23.0 1.2 2.5 0.8 0.75 0.70 5 2 0.1 0.85 3 1500 0.95 0.3 0.8 100 200 300 1 1 1000000 50 3 20 0.5 2.5 --benchmark 100

# 13. Rate-only output (for integration)

python3 python_files/6b_ml_client.py --features 25.0 24.0 23.0 1.2 2.5 0.8 0.75 0.70 5 2 0.1 0.85 3 1500 0.95 0.3 0.8 100 200 300 1 1 1000000 50 3 20 0.5 2.5 --format rate

# 14. Verbose debugging

python3 python_files/6b_ml_client.py --features 25.0 24.0 23.0 1.2 2.5 0.8 0.75 0.70 5 2 0.1 0.85 3 1500 0.95 0.3 0.8 100 200 300 1 1 1000000 50 3 20 0.5 2.5 --verbose

# 15. Shutdown server

python3 python_files/6b_ml_client.py --shutdown

# ================== EXAMPLE FEATURE VECTORS ==================

# Good WiFi conditions (example 1)

python3 python_files/6b_ml_client.py --features 30.0 29.0 28.0 0.5 1.8 0.9 0.95 0.90 10 1 0.05 0.95 1 500 0.98 0.1 0.9 50 100 150 0 1 5000000 20 1 80 0.2 1.2

# Poor WiFi conditions (example 2)

python3 python_files/6b_ml_client.py --features 10.0 12.0 11.0 -0.8 3.5 0.3 0.40 0.35 2 8 0.4 0.45 8 3000 0.6 0.8 0.4 500 800 1200 2 0 500000 200 15 20 0.8 5.5

# Marginal WiFi conditions (example 3)

python3 python_files/6b_ml_client.py --features 18.0 19.0 17.0 0.2 2.2 0.65 0.70 0.68 6 4 0.2 0.72 4 2000 0.8 0.5 0.7 200 400 600 1 1 2000000 100 5 40 0.4 3.0
