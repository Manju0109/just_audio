#import "KaraokeTap.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

typedef struct {
  float b0, b1, b2, a1, a2;
  float z1, z2;
} KaraokeBiquad;

static void KaraokeBiquadReset(KaraokeBiquad *filter) {
  if (!filter) return;
  filter->z1 = 0.f;
  filter->z2 = 0.f;
}

static void KaraokeBiquadSetBandPass(KaraokeBiquad *filter,
                                     float sampleRate,
                                     float frequency,
                                     float q) {
  if (!filter || sampleRate <= 0.f) return;
  const double omega = 2.0 * M_PI * frequency / (double)sampleRate;
  const double sinOmega = sin(omega);
  const double cosOmega = cos(omega);
  const double alpha = sinOmega / (2.0 * q);
  const double a0 = 1.0 + alpha;
  filter->b0 = (float)(alpha / a0);
  filter->b1 = 0.f;
  filter->b2 = (float)(-alpha / a0);
  filter->a1 = (float)(-2.0 * cosOmega / a0);
  filter->a2 = (float)((1.0 - alpha) / a0);
  KaraokeBiquadReset(filter);
}

static void KaraokeBiquadSetLowPass(KaraokeBiquad *filter,
                                    float sampleRate,
                                    float frequency,
                                    float q) {
  if (!filter || sampleRate <= 0.f) return;
  const double omega = 2.0 * M_PI * frequency / (double)sampleRate;
  const double sinOmega = sin(omega);
  const double cosOmega = cos(omega);
  const double alpha = sinOmega / (2.0 * q);
  const double a0 = 1.0 + alpha;
  const double a0Inv = 1.0 / a0;
  filter->b0 = (float)(((1.0 - cosOmega) * 0.5) * a0Inv);
  filter->b1 = (float)((1.0 - cosOmega) * a0Inv);
  filter->b2 = filter->b0;
  filter->a1 = (float)(-2.0 * cosOmega * a0Inv);
  filter->a2 = (float)((1.0 - alpha) * a0Inv);
  KaraokeBiquadReset(filter);
}

static float KaraokeBiquadProcess(KaraokeBiquad *filter, float input) {
  if (!filter) return input;
  float output = filter->b0 * input + filter->z1;
  filter->z1 = filter->b1 * input - filter->a1 * output + filter->z2;
  filter->z2 = filter->b2 * input - filter->a2 * output;
  return output;
}

typedef struct { __unsafe_unretained KaraokeTap *owner; } KaraokeTapContext;

@interface KaraokeTap () {
@public
  MTAudioProcessingTapRef _tap;
  KaraokeTapContext _ctx;
  float _uiLevel;
  float _level;
  float _smoothedLevel;
  float _sampleRate;
  bool _filtersConfigured;
  KaraokeBiquad _vocalBandFilter;
  KaraokeBiquad _presenceFilter;
  KaraokeBiquad _lowMidFilter;
  KaraokeBiquad _lowVocalFilter;
}
@end

static inline float clampf32(float v) { return fmaxf(-1.f, fminf(1.f, v)); }

static float KaraokeTap_MapUiLevel(float uiLevel) {
  if (uiLevel <= 0.f) return 0.f;
  if (uiLevel < 0.33f) {
    return (uiLevel / 0.33f) * 0.45f;
  } else if (uiLevel < 0.66f) {
    return 0.45f + ((uiLevel - 0.33f) / 0.33f) * 0.35f;
  } else {
    return 0.80f + ((uiLevel - 0.66f) / 0.34f) * 0.25f;
  }
}

static void KaraokeTap_ConfigureFilters(KaraokeTap *owner) {
  if (!owner) return;
  const float sampleRate = owner->_sampleRate > 0.f ? owner->_sampleRate : 44100.f;
  KaraokeBiquadSetBandPass(&owner->_vocalBandFilter, sampleRate, 900.f, 0.707f);
  KaraokeBiquadSetBandPass(&owner->_presenceFilter, sampleRate, 2600.f, 1.0f);
  KaraokeBiquadSetLowPass(&owner->_lowMidFilter, sampleRate, 220.f, 0.707f);
  KaraokeBiquadSetLowPass(&owner->_lowVocalFilter, sampleRate, 220.f, 0.707f);
  owner->_filtersConfigured = true;
}

static void KaraokeTap_Init(MTAudioProcessingTapRef tap,
                            void *clientInfo,
                            void **tapStorageOut)
{
  if (tapStorageOut != NULL) {
    *tapStorageOut = clientInfo;
  }
}

static void KaraokeTap_Process(MTAudioProcessingTapRef tap,
                               CMItemCount frameCount,
                               MTAudioProcessingTapFlags flags,
                               AudioBufferList *buffers,
                               CMItemCount *frameCountOut,
                               MTAudioProcessingTapFlags *flagsOut)
{
  (void)flags;
  KaraokeTapContext *ctx =
      (KaraokeTapContext *)MTAudioProcessingTapGetStorage(tap);
  if (!ctx) {
    if (frameCountOut) *frameCountOut = 0;
    if (flagsOut) *flagsOut = 0;
    return;
  }
  KaraokeTap *owner = ctx->owner;
  if (!owner) {
    if (frameCountOut) *frameCountOut = 0;
    if (flagsOut) *flagsOut = 0;
    return;
  }

  CMItemCount actualFrameCount = frameCount;
  CMTimeRange timeRange = kCMTimeRangeZero;
  OSStatus err = MTAudioProcessingTapGetSourceAudio(
      tap, frameCount, buffers, flagsOut, &timeRange, &actualFrameCount);
  if (err) {
    if (frameCountOut) *frameCountOut = 0;
    if (flagsOut) *flagsOut = 0;
    return;
  }
  if (frameCountOut) *frameCountOut = actualFrameCount;
  if (actualFrameCount == 0) {
    if (flagsOut) *flagsOut = 0;
    return;
  }

  if (!owner->_filtersConfigured) {
    KaraokeTap_ConfigureFilters(owner);
  }

  const float smoothing = 0.12f;
  float targetLevel = owner.level;
  owner->_smoothedLevel += smoothing * (targetLevel - owner->_smoothedLevel);
  float lvl = owner->_smoothedLevel;
  if (lvl <= 0.0005f) {
    return;
  }

  const float crossMix = 0.55f * lvl;
  const float sideGain = 1.0f + 0.35f * lvl;
  const float gainComp = 1.0f + 0.20f * lvl;
  const float highAttenuation = 1.0f - 0.90f * lvl;
  const float presenceBlend = 0.35f;

  if (buffers->mNumberBuffers == 0) {
    return;
  }

  UInt32 totalChannels = 0;
  for (UInt32 buf = 0; buf < buffers->mNumberBuffers; buf++) {
    totalChannels += buffers->mBuffers[buf].mNumberChannels;
  }
  if (totalChannels < 2) {
    return;
  }

  const UInt32 kInlineChannels = 8;
  float *channelBasesStack[kInlineChannels];
  UInt32 channelStridesStack[kInlineChannels];
  UInt32 channelFrameCountsStack[kInlineChannels];
  memset(channelBasesStack, 0, sizeof(channelBasesStack));
  memset(channelStridesStack, 0, sizeof(channelStridesStack));
  memset(channelFrameCountsStack, 0, sizeof(channelFrameCountsStack));

  float **channelBases = channelBasesStack;
  UInt32 *channelStrides = channelStridesStack;
  UInt32 *channelFrameCounts = channelFrameCountsStack;
  bool usedHeap = false;

  // Build a channel lookup table that works for both interleaved and planar
  // layouts, re-using stack storage for the common stereo case.
  if (totalChannels > kInlineChannels) {
    channelBases =
        (float **)calloc(totalChannels, sizeof(float *));
    channelStrides =
        (UInt32 *)calloc(totalChannels, sizeof(UInt32));
    channelFrameCounts =
        (UInt32 *)calloc(totalChannels, sizeof(UInt32));
    if (!channelBases || !channelStrides || !channelFrameCounts) {
      free(channelBases);
      free(channelStrides);
      free(channelFrameCounts);
      return;
    }
    usedHeap = true;
  }
  UInt32 channelCount = 0;

  for (UInt32 buf = 0; buf < buffers->mNumberBuffers; buf++) {
    float *data = (float *)buffers->mBuffers[buf].mData;
    UInt32 channels = buffers->mBuffers[buf].mNumberChannels;
    if (!data || channels == 0) {
      continue;
    }

    UInt32 framesPerBuffer = 0;
    if (channels == 1) {
      framesPerBuffer =
          buffers->mBuffers[buf].mDataByteSize / sizeof(float);
      channelBases[channelCount] = data;
      channelStrides[channelCount] = 1;
      channelFrameCounts[channelCount] = framesPerBuffer;
      channelCount++;
    } else {
      framesPerBuffer =
          buffers->mBuffers[buf].mDataByteSize / (sizeof(float) * channels);
      for (UInt32 ch = 0; ch < channels && channelCount < totalChannels;
           ch++) {
        channelBases[channelCount] = data + ch;
        channelStrides[channelCount] = channels;
        channelFrameCounts[channelCount] = framesPerBuffer;
        channelCount++;
      }
    }

    if (channelCount >= totalChannels) {
      break;
    }
  }

  if (channelCount < 2) {
    if (usedHeap) {
      free(channelBases);
      free(channelStrides);
      free(channelFrameCounts);
    }
    return;
  }

  // Apply center-cut attenuation to each stereo pair we discovered.
  for (UInt32 ch = 0; ch + 1 < channelCount; ch += 2) {
    float *left = channelBases[ch];
    float *right = channelBases[ch + 1];
    UInt32 leftStride = channelStrides[ch];
    UInt32 rightStride = channelStrides[ch + 1];
    UInt32 leftFrames = channelFrameCounts[ch];
    UInt32 rightFrames = channelFrameCounts[ch + 1];
    if (!left || !right || leftStride == 0 || rightStride == 0) {
      continue;
    }

    CMItemCount framesToProcess = actualFrameCount;
    if (leftFrames < framesToProcess) {
      framesToProcess = leftFrames;
    }
    if (rightFrames < framesToProcess) {
      framesToProcess = rightFrames;
    }
    if (framesToProcess == 0) {
      continue;
    }

    for (CMItemCount frame = 0; frame < framesToProcess; frame++) {
      float *lPtr = left + frame * leftStride;
      float *rPtr = right + frame * rightStride;
      float origL = *lPtr;
      float origR = *rPtr;
      float bleedReducedL = origL - crossMix * origR;
      float bleedReducedR = origR - crossMix * origL;
      float mid = (bleedReducedL + bleedReducedR) * 0.5f;
      float side = (bleedReducedL - bleedReducedR) * 0.5f;

      float lowMid = owner->_filtersConfigured
          ? KaraokeBiquadProcess(&owner->_lowMidFilter, mid)
          : mid * 0.5f;
      float highMid = mid - lowMid;

      float vocalEstimate = owner->_filtersConfigured
          ? KaraokeBiquadProcess(&owner->_vocalBandFilter, highMid)
          : highMid;
      float presenceEstimate = owner->_filtersConfigured
          ? KaraokeBiquadProcess(&owner->_presenceFilter, highMid)
          : highMid;
      float combinedVocal =
          (1.f - presenceBlend) * vocalEstimate + presenceBlend * presenceEstimate;

      float lowVocal = owner->_filtersConfigured
          ? KaraokeBiquadProcess(&owner->_lowVocalFilter, combinedVocal)
          : combinedVocal * 0.5f;
      float highVocal = combinedVocal - lowVocal;

      float cleanedHigh = (highMid - highVocal * lvl) * highAttenuation;
      float reducedMid = lowMid + cleanedHigh;
      float boostedSide = side * sideGain;

      float outL = (reducedMid + boostedSide) * gainComp;
      float outR = (reducedMid - boostedSide) * gainComp;
      *lPtr = clampf32(outL);
      *rPtr = clampf32(outR);
    }
  }

  if (usedHeap) {
    free(channelBases);
    free(channelStrides);
    free(channelFrameCounts);
  }
}

@implementation KaraokeTap

- (instancetype)init {
  if ((self = [super init])) {
    _uiLevel = 0.f;
    _level = 0.f;
    _smoothedLevel = 0.f;
    _sampleRate = 44100.f;
    _filtersConfigured = false;
    _ctx.owner = self;
    MTAudioProcessingTapCallbacks cb;
    cb.version = kMTAudioProcessingTapCallbacksVersion_0;
    cb.clientInfo = &_ctx;
    cb.init = KaraokeTap_Init;
    cb.finalize = NULL;
    cb.prepare = NULL;
    cb.unprepare = NULL;
    cb.process = KaraokeTap_Process;
    MTAudioProcessingTapCreate(kCFAllocatorDefault, &cb,
                               kMTAudioProcessingTapCreationFlag_PostEffects,
                               &_tap);
  }
  return self;
}

// Clamp level updates to [0, 1] and store atomically.
- (void)setLevel:(float)level {
  float clamped = fmaxf(0.f, fminf(1.f, level));
  @synchronized (self) {
    _uiLevel = clamped;
    _level = KaraokeTap_MapUiLevel(clamped);
    if (fabsf(_level) < 0.0001f) {
      _smoothedLevel = 0.f;
    }
  }
}

- (float)level {
  @synchronized (self) {
    return _level;
  }
}

- (void)updateSampleRate:(float)sampleRate {
  @synchronized (self) {
    if (sampleRate > 0.f) {
      _sampleRate = sampleRate;
    }
    KaraokeBiquadReset(&_vocalBandFilter);
    KaraokeBiquadReset(&_presenceFilter);
    KaraokeBiquadReset(&_lowMidFilter);
    KaraokeBiquadReset(&_lowVocalFilter);
    _filtersConfigured = false;
  }
}

// Surfaces the tap so AVMutableAudioMixInputParameters can attach it.
- (MTAudioProcessingTapRef)tap { return _tap; }

- (void)dealloc { if (_tap) CFRelease(_tap); }
@end
