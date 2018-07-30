# Audio
import models.Tacotron.hyperparams as hp
num_mels = 80
num_freq = 1024
sample_rate = 24000
frame_length_ms = 50.
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
max_iters = 200
griffin_lim_iters = 60
power = 1.5
outputs_per_step = hp.outputs_per_step

max_duration=5
max_length=100
