import socket
import random

# Helper to send a feature vector to the server and print response
def test_ml(features, host="localhost", port=8765):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall((features + "\n").encode('utf-8'))
    response = b""
    while True:
        data = s.recv(4096)
        if not data:
            break
        response += data
        if b"\n" in data:
            break
    s.close()
    print(f"Sent: {features}")
    print(f"Received: {response.decode('utf-8').strip()}")
    print("-"*70)

def make_case(rateIdx, phyRate, lastSnr, snrFast, snrSlow,
              shortSuccRatio, medSuccRatio, consecSuccess, consecFailure,
              severity, confidence, T1, T2, T3, decisionReason, packetSuccess,
              offeredLoad, queueLen, retryCount, channelWidth, mobilityMetric, snrVariance):
    # Compose 22 features as a space-separated string
    return f"{rateIdx} {phyRate} {lastSnr} {snrFast} {snrSlow} {shortSuccRatio} {medSuccRatio} {consecSuccess} {consecFailure} {severity} {confidence} {T1} {T2} {T3} {decisionReason} {packetSuccess} {offeredLoad} {queueLen} {retryCount} {channelWidth} {mobilityMetric} {snrVariance}"

test_cases = []

# 1. All rates (0-7), SNR increases, success/failure alternate, confidence varies
for rate in range(8):
    test_cases.append(make_case(
        rate,
        1e6*(rate+1),
        10+rate*5,
        10+rate*5,
        10+rate*5,
        0.1*rate,
        0.15*rate,
        rate*5,
        max(0, 35-rate*5),
        min(1.0, 0.2+rate*0.1),
        max(0.1, 1.0-rate*0.1),
        rate*2,
        rate*3,
        rate*4,
        rate%4,
        int(rate%2==0),
        10+rate,
        5+rate,
        rate,
        20,
        min(1.0, rate*0.12),
        1+rate*3
    ))

# 2. Edge: all zeros
test_cases.append("0 "*21 + "0")

# 3. Edge: all max values
test_cases.append(make_case(
    7, 54000000, 50, 50, 50, 1, 1, 100, 100, 1, 1, 100, 100, 100, 100, 1, 100, 100, 100, 20, 1, 100
))

# 4. Edge: all min values
test_cases.append(make_case(
    0, 1000000, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
))

# 5. High mobility, low SNR, high retry
test_cases.append(make_case(
    3, 12000000, 7, 8, 9, 0.3, 0.2, 2, 30, 0.6, 0.4, 12, 15, 18, 2, 0, 15, 10, 7, 40, 0.95, 30
))

# 6. Emergency context: low SNR, many failures
test_cases.append(make_case(
    0, 1000000, 6, 5, 4, 0.1, 0.1, 0, 100, 1, 0.1, 1, 1, 1, 1, 0, 8, 3, 5, 20, 0.1, 5
))

# 7. Very stable, high confidence, max throughput
test_cases.append(make_case(
    7, 54000000, 50, 50, 50, 1, 1, 80, 0, 0.0, 1.0, 100, 100, 100, 0, 1, 100, 0, 0, 20, 0.0, 2
))

# 8. Low queue, low success, moderate SNR
test_cases.append(make_case(
    2, 3000000, 22, 23, 21, 0.2, 0.3, 1, 16, 0.3, 0.5, 8, 10, 12, 1, 0, 12, 2, 2, 20, 0.25, 6
))

# 9. High queue, high retry, high variance
test_cases.append(make_case(
    4, 20000000, 35, 34, 36, 0.7, 0.8, 30, 2, 0.4, 0.6, 44, 66, 88, 3, 1, 22, 15, 12, 20, 0.7, 55
))

# 10. Negative values (should clamp)
test_cases.append(make_case(
    -1, -1000000, -10, -10, -10, -0.1, -0.5, -50, -50, -1, -1, -10, -10, -10, -1, -1, -10, -10, -10, -10, -1, -10
))

# 11. Over-max values (should clamp)
test_cases.append(make_case(
    8, 60000000, 90, 90, 90, 1.5, 1.2, 200, 200, 2, 2, 200, 200, 200, 200, 2, 200, 200, 200, 40, 2, 200
))

# 12. Randomized feature values (repeatable seed)
random.seed(42)
for _ in range(4):
    test_cases.append(" ".join(str(round(random.uniform(0, 50), 2)) for _ in range(22)))

# 13. Wrong feature count (23, 21)
test_cases.append("1 " * 23)
test_cases.append("2 " * 21)

# 14. Fluctuating SNR, random rates, random success
for rate in [0, 3, 5, 7]:
    test_cases.append(make_case(
        rate,
        random.choice([1000000, 6000000, 54000000]),
        random.uniform(5, 50),
        random.uniform(5, 50),
        random.uniform(5, 50),
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.randint(0, 100),
        random.randint(0, 100),
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.randint(0, 100),
        random.randint(0, 100),
        random.randint(0, 100),
        random.randint(0, 3),
        random.randint(0, 1),
        random.randint(0, 100),
        random.randint(0, 100),
        random.randint(0, 20),
        random.choice([20, 40]),
        random.uniform(0, 1),
        random.uniform(0, 100)
    ))

for case in test_cases:
    case = case.strip()
    test_ml(case)