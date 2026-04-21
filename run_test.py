"""Quick sanity: 1 training step with reduced batch, measures time."""
import os, sys, time

os.environ.update({
    'CC': r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe',
    'RUN_ID': 'fused_1step',
    'SEED': '1337',
    'ITERATIONS': '1',
    'MAX_WALLCLOCK_SECONDS': '0',
    'TRAIN_BATCH_TOKENS': '4096',  # Minimum: 2 seqs * 2048 tokens
    'VAL_LOSS_EVERY': '99',
    'TRAIN_LOG_EVERY': '1',
    'WARMUP_STEPS': '0',
})

LOG = open('run_test_output.txt', 'w', buffering=1)
class Tee:
    def __init__(self, *s): self.s = s
    def write(self, t):
        for x in self.s: x.write(t); x.flush()
    def flush(self):
        for x in self.s: x.flush()
sys.stdout = Tee(sys.__stdout__, LOG)
sys.stderr = Tee(sys.__stderr__, LOG)

print("Starting import...")
import train_gpt
print(f"_USE_TRITON_MLP = {train_gpt._USE_TRITON_MLP}")
print("Starting main()...")
t0 = time.time()
try:
    train_gpt.main()
except Exception as e:
    print(f"EXCEPTION after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    print(f"Total time: {time.time()-t0:.1f}s")
    LOG.close()
