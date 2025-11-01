#import <Foundation/Foundation.h>
@import AVFoundation;

@interface KaraokeTap : NSObject
@property (atomic, assign) float level; // 0.0 = no reduction, 1.0 = max reduction
@property (nonatomic, readonly) MTAudioProcessingTapRef tap;
- (void)updateSampleRate:(float)sampleRate;
- (instancetype)init;
@end
