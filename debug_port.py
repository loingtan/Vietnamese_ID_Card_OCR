#!/usr/bin/env python3
"""Debug script to test port finding functionality."""

import socket
import time


def find_available_port_debug(start_port: int, max_attempts: int = 100) -> int:
    """
    Debug version of find_available_port with detailed logging.
    """
    print(f"Starting port search from {start_port}")

    for i, port in enumerate(range(start_port, start_port + max_attempts)):
        print(f"Attempt {i+1}/{max_attempts}: Testing port {port}")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Set socket options for better Windows compatibility
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.settimeout(5.0)  # Add timeout
                s.bind(('localhost', port))
                print(f"✓ Found available port: {port}")
                return port
        except OSError as e:
            print(f"✗ Port {port} not available: {e}")
            continue
        except Exception as e:
            print(f"✗ Unexpected error on port {port}: {e}")
            continue

    raise RuntimeError(
        f"Could not find an available port after {max_attempts} attempts")


def test_specific_ports():
    """Test specific ports that might be problematic."""
    test_ports = [8080, 8000, 8081, 8001, 9000, 9001]

    for port in test_ports:
        print(f"\nTesting port {port}:")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.settimeout(2.0)
                s.bind(('localhost', port))
                print(f"  ✓ Port {port} is available")
        except OSError as e:
            print(f"  ✗ Port {port} is not available: {e}")
        except Exception as e:
            print(f"  ✗ Error testing port {port}: {e}")


if __name__ == "__main__":
    print("=== Port Availability Debug ===")

    # Test specific ports first
    test_specific_ports()

    print("\n=== Testing Port Finding Function ===")

    try:
        # Test with default ports
        print("Testing port 8080:")
        port = find_available_port_debug(8080)
        print(f"Found port: {port}")

        print("\nTesting port 8000:")
        metrics_port = find_available_port_debug(8000)
        print(f"Found metrics port: {metrics_port}")

    except Exception as e:
        print(f"Error: {e}")
