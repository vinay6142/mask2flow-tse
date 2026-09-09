Mask2Flow-TSE -- listening samples for manual verification
============================================================================

WHY THIS EXISTS
----------------
The project's automated metrics (mel-MSE %, SI-SDR, WavLM cosine similarity,
corpus-wide accuracy/EER) are all proxies for "did we extract the right
speaker's voice, and does it sound right." This directory has the actual
audio so you can judge that yourself, for 2/3/4-total-speaker mixtures.

FOLDER LAYOUT
-------------
  2speakers/sample_00/, sample_01/, ...
  3speakers/sample_00/, sample_01/, ...
  4speakers/sample_00/, sample_01/, ...

Each sample_NN/ folder has 5 files + info.json:
  mixture.wav       what a microphone would pick up (target + all interferers)
  reference.wav     the enrollment clip the system was told to extract
                     ("find THIS voice in the mixture") -- trailing silence
                     already trimmed off
  extracted.wav     *** the actual system output -- listen to this one for
                     "did it get the right voice" ***
  target_clean.wav  ground-truth clean target, vocoded from ITS OWN real
                     mel (not predicted) -- the ceiling / oracle
  stage1_only.wav   masking-only output, before flow matching -- useful to
                     hear how much Stage 2 changes things
  info.json         speaker IDs, SNRs, and every metric for this exact
                     sample (mel %, SI-SDR, cosine similarity)

SUGGESTED LISTENING ORDER
--------------------------
1. reference.wav       -- learn what the target voice sounds like
2. mixture.wav          -- hear how buried it is in the mixture
3. extracted.wav        -- did the system pull the right voice out?
4. target_clean.wav     -- compare against the true target
5. stage1_only.wav      -- optional, hear the masking-only intermediate

TWO SPECIFIC THINGS TO LISTEN FOR
-----------------------------------
(A) "Metallic" vocoder quality. Compare extracted.wav against
    target_clean.wav. Both go through the SAME vocoder.
      - If target_clean.wav ALSO sounds metallic: that's a vocoder quality
        issue, not an extraction issue -- the vocoder itself struggles even
        with a real, undistorted mel spectrogram.
      - If target_clean.wav sounds clean/natural but extracted.wav sounds
        metallic: that's specifically Stage 2's mel prediction being
        imperfect (a distribution mismatch from what the vocoder was
        trained on), not a vocoder training problem.

(B) Low speaker-similarity despite a correct-sounding voice. info.json's
    speaker_sim_ref_vs_extracted is the cosine similarity between the
    reference embedding and the extracted output's embedding (same metric
    family as the project's accuracy/EER numbers). If the voice sounds
    right to you but this number seems low, check
    speaker_sim_ref_vs_target_clean first -- that's the SAME reference
    compared against the true clean target (same speaker, different
    utterance), and should be the highest number in the file. If even
    THAT number looks lower than expected, the reference clip itself may
    still be short/atypical for this speaker (trailing silence is now
    trimmed, but a reference clip that's just inherently a-typical for a
    speaker's voice is a separate, real limitation of any embedding-based
    metric, not a bug).

METRICS DEFINITIONS (info.json)
---------------------------------
  mel_s2_vs_s1_pct               mel-domain %-improvement, Stage 2 vs Stage 1
  sisdr_*_vs_target              SI-SDR (dB) of that signal against the true
                                  clean target -- higher is better
  speaker_sim_ref_vs_target_clean  reference vs. true target (sanity ceiling)
  speaker_sim_ref_vs_mixture       reference vs. raw mixture (do-nothing baseline)
  speaker_sim_ref_vs_stage1        reference vs. masking-only output
  speaker_sim_ref_vs_extracted     reference vs. the actual system output
                                    (THIS is the number reported throughout
                                    the project as "accuracy"/"similarity")
