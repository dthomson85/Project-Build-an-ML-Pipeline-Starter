import torch
import time
import math

def sieve_of_eratosthenes_gpu(n, batch_size=10_000_000):
    """
    GPU-accelerated Sieve of Eratosthenes using PyTorch
    
    Args:
        n: Find all primes up to n
        batch_size: Process in batches to manage GPU memory
    """
    print(f"Finding primes up to {n:,} using GPU...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    start_time = time.time()
    
    # Create sieve on GPU (True = potentially prime)
    is_prime = torch.ones(n + 1, dtype=torch.bool, device='cuda')
    is_prime[0] = is_prime[1] = False  # 0 and 1 are not prime
    
    # Only need to check up to sqrt(n)
    limit = int(math.sqrt(n)) + 1
    
    print("Running Sieve of Eratosthenes on GPU...")
    
    for i in range(2, limit):
        if i % 10000 == 0:
            print(f"Progress: {i:,} / {limit:,}")
        
        if is_prime[i].item():
            # Mark all multiples of i as not prime
            # Use GPU vectorization for speed
            multiples = torch.arange(i*i, n + 1, i, device='cuda')
            is_prime[multiples] = False
    
    # Count primes
    prime_count = is_prime.sum().item()
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Found {prime_count:,} prime numbers up to {n:,}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Rate: {n/elapsed:,.0f} numbers/second")
    print(f"{'='*50}\n")
    
    return is_prime, prime_count

def get_primes_list(is_prime, max_results=100):
    """Extract actual prime numbers from sieve (first max_results)"""
    indices = torch.nonzero(is_prime).squeeze()
    if len(indices) > max_results:
        return indices[:max_results].cpu().tolist()
    return indices.cpu().tolist()

if __name__ == "__main__":
    # Test with smaller number first
    print("WARM-UP TEST: Finding primes up to 10 million...")
    is_prime_small, count_small = sieve_of_eratosthenes_gpu(10_000_000)
    
    # Show first 50 primes
    first_primes = get_primes_list(is_prime_small, 50)
    print(f"First 50 primes: {first_primes}\n")
    
    # Now the big one - 1 billion
    print("\n" + "="*50)
    print("MAIN CALCULATION: 1 BILLION")
    print("="*50 + "\n")
    
    is_prime_big, count_big = sieve_of_eratosthenes_gpu(1_000_000_000)
    
    # Show some large primes
    print("Last 20 primes found:")
    all_primes = torch.nonzero(is_prime_big).squeeze()
    last_primes = all_primes[-20:].cpu().tolist()
    for p in last_primes:
        print(f"  {p:,}")
    
    print("\n🎉 GPU calculation complete!")
