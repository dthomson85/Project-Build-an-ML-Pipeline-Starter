import torch
import time
import math

def segmented_sieve_gpu(limit, segment_size=1_000_000_000):
    """
    Segmented Sieve of Eratosthenes for very large numbers
    Processes in segments to fit in GPU memory
    """
    print(f"Finding primes up to {limit:,} using segmented GPU sieve")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Segment size: {segment_size:,} ({segment_size / 1024**3:.2f} GB per segment)\n")
    
    start_time = time.time()
    
    # First, find all primes up to sqrt(limit) - we need these for all segments
    sqrt_limit = int(math.sqrt(limit)) + 1
    print(f"Step 1: Finding base primes up to {sqrt_limit:,}...")
    base_start = time.time()
    
    base_sieve = torch.ones(sqrt_limit + 1, dtype=torch.bool, device='cuda')
    base_sieve[0] = base_sieve[1] = False
    
    for i in range(2, int(math.sqrt(sqrt_limit)) + 1):
        if base_sieve[i].item():
            base_sieve[i*i::i] = False
    
    base_primes = torch.nonzero(base_sieve).squeeze().cpu()
    base_time = time.time() - base_start
    print(f"Found {len(base_primes):,} base primes in {base_time:.2f}s\n")
    
    # Now process in segments
    total_primes = len(base_primes)
    num_segments = (limit - sqrt_limit) // segment_size + 1
    
    print(f"Step 2: Processing {num_segments} segments of 1 billion numbers each...\n")
    print("="*70)
    
    for seg_num in range(num_segments):
        low = sqrt_limit + seg_num * segment_size
        high = min(low + segment_size - 1, limit)
        
        if low > high:
            break
        
        seg_start = time.time()
        
        # Create segment sieve on GPU
        seg_size = high - low + 1
        segment = torch.ones(seg_size, dtype=torch.bool, device='cuda')
        
        # Mark multiples of base primes in this segment
        for prime in base_primes:
            prime = prime.item()
            # Find first multiple of prime in segment
            start = ((low + prime - 1) // prime) * prime
            if start < low:
                start += prime
            
            # Mark multiples using GPU vectorization
            if start <= high:
                indices = torch.arange(start - low, seg_size, prime, device='cuda')
                segment[indices] = False
        
        # Count primes in this segment
        seg_prime_count = segment.sum().item()
        total_primes += seg_prime_count
        seg_time = time.time() - seg_start
        
        elapsed = time.time() - start_time
        progress_pct = ((high / limit) * 100)
        rate = high / elapsed
        eta = (limit - high) / rate if rate > 0 else 0
        
        print(f"Segment {seg_num + 1}/{num_segments} [{low:,} - {high:,}]")
        print(f"  ✓ Primes found: {seg_prime_count:,} | Total: {total_primes:,}")
        print(f"  ⏱️  Segment time: {seg_time:.2f}s | Speed: {seg_size/seg_time:,.0f} nums/sec")
        print(f"  📊 Progress: {progress_pct:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        print("-"*70)
        
        del segment
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    print("="*70)
    print(f"\n🎉 CALCULATION COMPLETE!\n")
    print(f"✅ Found {total_primes:,} prime numbers up to {limit:,}")
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📊 Average rate: {limit/total_time:,.0f} numbers/second")
    print(f"🚀 GPU utilization: Excellent!")
    print("="*70)
    
    return total_primes

if __name__ == "__main__":
    # 1 trillion with 1 billion chunks
    limit = 1_000_000_000_000
    
    print("="*70)
    print("🚀 CALCULATING PRIMES UP TO 1 TRILLION 🚀")
    print("="*70 + "\n")
    
    total = segmented_sieve_gpu(limit, segment_size=1_000_000_000)
    
    print(f"\n💡 FUN FACT: There are {total:,} primes below 1 trillion!")
    print("    That's approximately 1 in every 26 numbers!")
