# ================== SERVER COMMANDS ==================

## 1. Create configuration file

```bash
python3 python_files/6a_ml_inference_server.py --create-config
```

## 2. Start enhanced server (basic)

```bash
python3 python_files/6a_ml_inference_server.py --config server_config.json
```

## 3. Start server with custom port and debug logging

```bash
python3 python_files/6a_ml_inference_server.py --config server_config.json --port 9000 --log-level DEBUG
```

## 4. Background server with logging

```bash
python3 python_files/6a_ml_inference_server.py --config server_config.json > server.log 2>&1 &
```

# ================== CLIENT COMMANDS ==================

## 5. Get server info

```bash
python3 python_files/6b_ml_client.py --info
```

## 6. List available models

```bash
python3 python_files/6b_ml_client.py --models
```

## 7. Test server connectivity

```bash
python3 python_files/6b_ml_client.py --ping
```

## 8. Get server statistics

```bash
python3 python_files/6b_ml_client.py --stats --format json
```

## 9. Single prediction with oracle_balanced model (21 safe features)

```bash
python3 python_files/6b_ml_client.py --features 30.0 29.0 28.0 0.5 1.8 0.9 5.0 0.95 0.90 10 1 0.05 0.95 1 50 0.98 0.1 0.9 20 0.2 1.2 --model oracle_balanced
```

## 10. Prediction with compact output format

```bash
python3 python_files/6b_ml_client.py --features 30.0 29.0 28.0 0.5 1.8 0.9 5.0 0.95 0.90 10 1 0.05 0.95 1 50 0.98 0.1 0.9 20 0.2 1.2 --format compact
```

## 11. Load features from file

```bash
echo "30.0 29.0 28.0 0.5 1.8 0.9 5.0 0.95 0.90 10 1 0.05 0.95 1 50 0.98 0.1 0.9 20 0.2 1.2" > features.txt
python3 python_files/6b_ml_client.py --features-file features.txt --format detailed
```

## 12. Benchmark performance (100 predictions)

```bash
python3 python_files/6b_ml_client.py --features 30.0 29.0 28.0 0.5 1.8 0.9 5.0 0.95 0.90 10 1 0.05 0.95 1 50 0.98 0.1 0.9 20 0.2 1.2 --benchmark 100
```

## 13. Rate-only output (for integration)

```bash
python3 python_files/6b_ml_client.py --features 30.0 29.0 28.0 0.5 1.8 0.9 5.0 0.95 0.90 10 1 0.05 0.95 1 50 0.98 0.1 0.9 20 0.2 1.2 --format rate
```

## 14. Verbose debugging

```bash
python3 python_files/6b_ml_client.py --features 30.0 29.0 28.0 0.5 1.8 0.9 5.0 0.95 0.90 10 1 0.05 0.95 1 50 0.98 0.1 0.9 20 0.2 1.2 --verbose
```

## 15. Shutdown server

```bash
python3 python_files/6b_ml_client.py --shutdown
```

# ================== EXAMPLE FEATURE VECTORS (21 FEATURES) ==================

## Good WiFi conditions

```bash
python3 python_files/6b_ml_client.py --features 35.0 34.0 33.0 0.6 2.2 0.95 2.0 0.99 0.98 18 0 0.01 0.97 2 100 0.99 0.05 0.98 40 0.01 0.7 --model oracle_balanced
```

## Poor WiFi conditions

```bash
python3 python_files/6b_ml_client.py --features 8.0 9.0 10.0 -0.9 5.5 0.3 7.0 0.42 0.39 1 7 0.7 0.41 9 400 0.7 0.6 0.5 200 0.8 3.0 --model oracle_balanced
```

## Marginal WiFi conditions

```bash
python3 python_files/6b_ml_client.py --features 17.0 18.0 16.0 0.3 2.1 0.7 4.0 0.75 0.73 4 3 0.21 0.78 5 150 0.85 0.3 0.8 60 0.3 1.8 --model oracle_balanced
```
