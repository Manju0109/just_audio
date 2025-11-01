package com.ryanheise.just_audio;

import androidx.media3.common.C;
import androidx.media3.common.audio.AudioProcessor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Mid/Side karaoke processor that attenuates the centre channel (vocals).
 */
final class KaraokeAudioProcessor implements AudioProcessor {

    private static final ByteBuffer EMPTY_BUFFER =
        ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder());
    private static final float EPSILON = 0.0001f;
    private static final float SMOOTH_ALPHA = 0.12f;
    private static final float PRESENCE_BLEND = 0.35f;
    private static final float LOW_BAND_FREQ = 220f;
    private static final float LOW_BAND_Q = 0.707f;
    private static final float VOCAL_BAND_FREQ = 900f;
    private static final float VOCAL_BAND_Q = 0.707f;
    private static final float PRESENCE_FREQ = 2600f;
    private static final float PRESENCE_Q = 1.0f;

    private volatile float uiLevel = 0f;
    private volatile float targetLevel = 0f;
    private float smoothedLevel = 0f;
    private AudioFormat inputAudioFormat = AudioFormat.NOT_SET;
    private AudioFormat outputAudioFormat = AudioFormat.NOT_SET;
    private ByteBuffer buffer = EMPTY_BUFFER;
    private ByteBuffer outputBuffer = EMPTY_BUFFER;
    private boolean inputEnded = false;
    private final BiquadFilter vocalBandFilter = new BiquadFilter();
    private final BiquadFilter presenceFilter = new BiquadFilter();
    private final BiquadFilter lowMidPreserver = new BiquadFilter();
    private final BiquadFilter lowVocalFilter = new BiquadFilter();
    private boolean filtersConfigured = false;

    void setLevel(float newLevel) {
        uiLevel = Math.max(0f, Math.min(1f, newLevel));
        targetLevel = mapUiLevel(uiLevel);
    }

    @Override
    public boolean isActive() {
        return (targetLevel > EPSILON || smoothedLevel > EPSILON)
            && inputAudioFormat.encoding != C.ENCODING_INVALID;
    }

    public AudioFormat getOutputFormat() {
        return outputAudioFormat;
    }

    @Override
    public AudioFormat configure(AudioFormat inputFormat) throws UnhandledAudioFormatException {
        if (inputFormat.encoding != C.ENCODING_PCM_16BIT || inputFormat.channelCount != 2) {
            throw new UnhandledAudioFormatException(inputFormat);
        }
        inputAudioFormat = inputFormat;
        outputAudioFormat = inputFormat;
        configureFilters(inputFormat.sampleRate);
        return outputAudioFormat;
    }

    @Override
    public void queueInput(ByteBuffer inputBuffer) {
        int bytes = inputBuffer.remaining();
        if (bytes == 0) {
            return;
        }
        ensureCapacity(bytes);

        ByteBuffer inBuffer = inputBuffer.order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer outBuffer = buffer;
        outBuffer.clear();
        outBuffer.limit(bytes);
        outBuffer.order(ByteOrder.LITTLE_ENDIAN);

        smoothedLevel += SMOOTH_ALPHA * (targetLevel - smoothedLevel);
        final float mixLevel = smoothedLevel;
        final float crossMix = 0.55f * mixLevel;
        final float sideGain = 1f + 0.35f * mixLevel;
        final float gainComp = 1f + 0.2f * mixLevel;
        final float highAttenuation = 1f - 0.9f * mixLevel;
        final float fallbackLevel = 0.0005f;

        while (inBuffer.hasRemaining()) {
            short leftSample = inBuffer.getShort();
            short rightSample = inBuffer.getShort();

            final float left = leftSample;
            final float right = rightSample;

            if (mixLevel <= fallbackLevel) {
                outBuffer.putShort(leftSample);
                outBuffer.putShort(rightSample);
                continue;
            }

            final float bleedReducedLeft = left - crossMix * right;
            final float bleedReducedRight = right - crossMix * left;
            final float mid = (bleedReducedLeft + bleedReducedRight) * 0.5f;
            final float side = (bleedReducedLeft - bleedReducedRight) * 0.5f;

            float lowMid = filtersConfigured
                ? lowMidPreserver.process(mid)
                : mid * 0.5f;
            float highMid = mid - lowMid;

            float vocalEstimate = filtersConfigured
                ? vocalBandFilter.process(highMid)
                : highMid;
            float presenceEstimate = filtersConfigured
                ? presenceFilter.process(highMid)
                : highMid;
            float combinedVocal = (1f - PRESENCE_BLEND) * vocalEstimate
                + PRESENCE_BLEND * presenceEstimate;

            float lowVocal = filtersConfigured
                ? lowVocalFilter.process(combinedVocal)
                : combinedVocal * 0.5f;
            float highVocal = combinedVocal - lowVocal;

            float cleanedHigh = (highMid - highVocal * mixLevel) * highAttenuation;
            float reducedMid = lowMid + cleanedHigh;
            float boostedSide = side * sideGain;

            short outputLeft = (short) clamp16((reducedMid + boostedSide) * gainComp);
            short outputRight = (short) clamp16((reducedMid - boostedSide) * gainComp);

            outBuffer.putShort(outputLeft);
            outBuffer.putShort(outputRight);
        }

        inputBuffer.position(inputBuffer.limit());
        outBuffer.flip();
        outputBuffer = outBuffer;
    }

    @Override
    public void queueEndOfStream() {
        inputEnded = true;
    }

    @Override
    public long getDurationAfterProcessorApplied(long inputDurationUs) {
        return inputDurationUs;
    }

    @Override
    public ByteBuffer getOutput() {
        ByteBuffer output = outputBuffer;
        outputBuffer = EMPTY_BUFFER;
        return output;
    }

    @Override
    public boolean isEnded() {
        return inputEnded && !outputBuffer.hasRemaining();
    }

    @Override
    public void flush() {
        outputBuffer = EMPTY_BUFFER;
        inputEnded = false;
        vocalBandFilter.reset();
        presenceFilter.reset();
        lowMidPreserver.reset();
        lowVocalFilter.reset();
        smoothedLevel = targetLevel;
    }

    @Override
    public void reset() {
        flush();
        inputAudioFormat = AudioFormat.NOT_SET;
        outputAudioFormat = AudioFormat.NOT_SET;
        buffer = EMPTY_BUFFER;
        filtersConfigured = false;
        smoothedLevel = 0f;
        targetLevel = 0f;
        uiLevel = 0f;
    }

    private void ensureCapacity(int requiredCapacity) {
        if (buffer.capacity() < requiredCapacity) {
            buffer = ByteBuffer.allocateDirect(requiredCapacity).order(ByteOrder.LITTLE_ENDIAN);
        }
    }

    private void configureFilters(int sampleRate) {
        if (sampleRate <= 0) {
            filtersConfigured = false;
            return;
        }
        vocalBandFilter.setBandPass(sampleRate, VOCAL_BAND_FREQ, VOCAL_BAND_Q);
        presenceFilter.setBandPass(sampleRate, PRESENCE_FREQ, PRESENCE_Q);
        lowMidPreserver.setLowPass(sampleRate, LOW_BAND_FREQ, LOW_BAND_Q);
        lowVocalFilter.setLowPass(sampleRate, LOW_BAND_FREQ, LOW_BAND_Q);
        filtersConfigured = true;
    }

    private static int clamp16(float sample) {
        if (sample > 32767f) {
            return 32767;
        }
        if (sample < -32768f) {
            return -32768;
        }
        return Math.round(sample);
    }

    private static float mapUiLevel(float uiLevel) {
        if (uiLevel <= 0f) {
            return 0f;
        }
        if (uiLevel < 0.33f) {
            return (uiLevel / 0.33f) * 0.45f;
        } else if (uiLevel < 0.66f) {
            return 0.45f + ((uiLevel - 0.33f) / 0.33f) * 0.35f;
        } else {
            return 0.80f + ((uiLevel - 0.66f) / 0.34f) * 0.25f;
        }
    }

    private static final class BiquadFilter {
        private float b0, b1, b2, a1, a2;
        private float z1 = 0f;
        private float z2 = 0f;

        void setBandPass(int sampleRate, float frequency, float q) {
            double omega = 2.0 * Math.PI * frequency / (double) sampleRate;
            double sin = Math.sin(omega);
            double cos = Math.cos(omega);
            double alpha = sin / (2.0 * q);

            double a0 = 1.0 + alpha;
            b0 = (float) (alpha / a0);
            b1 = 0f;
            b2 = (float) (-alpha / a0);
            a1 = (float) (-2.0 * cos / a0);
            a2 = (float) ((1.0 - alpha) / a0);
            reset();
        }

        void setLowPass(int sampleRate, float frequency, float q) {
            double omega = 2.0 * Math.PI * frequency / (double) sampleRate;
            double sin = Math.sin(omega);
            double cos = Math.cos(omega);
            double alpha = sin / (2.0 * q);

            double a0 = 1.0 + alpha;
            double a0Inv = 1.0 / a0;
            b0 = (float) ((1.0 - cos) * 0.5 * a0Inv);
            b1 = (float) ((1.0 - cos) * a0Inv);
            b2 = b0;
            a1 = (float) (-2.0 * cos * a0Inv);
            a2 = (float) ((1.0 - alpha) * a0Inv);
            reset();
        }

        float process(float input) {
            float output = b0 * input + z1;
            z1 = b1 * input - a1 * output + z2;
            z2 = b2 * input - a2 * output;
            return output;
        }

        void reset() {
            z1 = 0f;
            z2 = 0f;
        }
    }
}
