# Remotion 技术参考

> Agent 在 `compose` 阶段选择 Remotion 渲染时的参考。
> Remotion 适合需要精确控制文字动画、Kinetic Typography 的旅行 Vlog。

## 1. 环境准备

### 1.1 初始化 Remotion 项目

```bash
npx create-video@latest travel-remotion
cd travel-remotion
npm install
```

### 1.2 目录结构

```
travel-remotion/
├── src/
│   ├── Root.tsx          # 注册所有 composition
│   ├── TravelVlog.tsx    # 旅行 Vlog 主组件
│   ├── components/
│   │   ├── Clip.tsx       # 单个素材片段
│   │   ├── Subtitle.tsx   # 字幕组件
│   │   ├── Transition.tsx # 转场组件
│   │   └── BGM.tsx        # 背景音乐
│   └── utils/
│       └── timeline.ts    # 时间线工具
├── public/
│   ├── clips/             # 素材片段
│   └── audio/             # BGM / 旁白
└── remotion.config.ts
```

## 2. 核心 Composition

### 2.1 TravelVlog.tsx

```tsx
import {Composition, AbsoluteFill, Sequence, Audio} from 'remotion';
import {Clip} from './components/Clip';
import {Subtitle} from './components/Subtitle';
import {Transition} from './components/Transition';

interface TimelineItem {
  order: number;
  source_path: string;
  in_sec: number;
  out_sec: number;
  duration_sec: number;
  transition_in: string;
  subtitle?: string;
  narration_text?: string;
}

export const TravelVlog: React.FC<{timeline: TimelineItem[]; bgm?: string}> = ({timeline, bgm}) => {
  let currentFrame = 0;
  const fps = 30;

  return (
    <AbsoluteFill style={{backgroundColor: 'black'}}>
      {timeline.map((item, index) => {
        const startFrame = currentFrame;
        const durationFrames = Math.round(item.duration_sec * fps);
        currentFrame += durationFrames;

        return (
          <Sequence key={index} from={startFrame} durationInFrames={durationFrames}>
            <Clip
              src={item.source_path}
              inSec={item.in_sec}
              outSec={item.out_sec}
              transition={item.transition_in}
            />
            {item.subtitle && (
              <Subtitle text={item.subtitle} startFrame={0} durationFrames={durationFrames} />
            )}
          </Sequence>
        );
      })}

      {bgm && <Audio src={bgm} volume={0.4} />}
    </AbsoluteFill>
  );
};
```

### 2.2 Clip.tsx

```tsx
import {AbsoluteFill, OffthreadVideo, interpolate, useCurrentFrame} from 'remotion';

interface ClipProps {
  src: string;
  inSec: number;
  outSec: number;
  transition: string;
}

export const Clip: React.FC<ClipProps> = ({src, inSec, outSec, transition}) => {
  const frame = useCurrentFrame();
  const fps = 30;

  // 转场效果
  let style: React.CSSProperties = {};
  if (transition === 'fade') {
    const opacity = interpolate(frame, [0, 15], [0, 1], {extrapolateRight: 'clamp'});
    style = {opacity};
  } else if (transition === 'zoom') {
    const scale = interpolate(frame, [0, 30], [1.1, 1.0], {extrapolateRight: 'clamp'});
    style = {transform: `scale(${scale})`};
  }

  return (
    <AbsoluteFill style={style}>
      <OffthreadVideo
        src={src}
        startFrom={Math.round(inSec * fps)}
        endAt={Math.round(outSec * fps)}
        style={{width: '100%', height: '100%', objectFit: 'cover'}}
      />
    </AbsoluteFill>
  );
};
```

### 2.3 Subtitle.tsx

```tsx
import {AbsoluteFill, interpolate, useCurrentFrame, spring} from 'remotion';

interface SubtitleProps {
  text: string;
  startFrame: number;
  durationFrames: number;
}

export const Subtitle: React.FC<SubtitleProps> = ({text, durationFrames}) => {
  const frame = useCurrentFrame();

  // 弹入效果
  const scale = spring({
    frame,
    fps: 30,
    config: {damping: 12, stiffness: 200},
  });

  // 淡出效果
  const opacity = interpolate(
    frame,
    [durationFrames - 10, durationFrames],
    [1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}
  );

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'flex-end',
        alignItems: 'center',
        paddingBottom: '120px',
      }}
    >
      <div
        style={{
          fontSize: '48px',
          fontWeight: 700,
          color: 'white',
          textShadow: '0 2px 8px rgba(0,0,0,0.8)',
          transform: `scale(${scale})`,
          opacity,
          fontFamily: 'Noto Sans SC, sans-serif',
        }}
      >
        {text}
      </div>
    </AbsoluteFill>
  );
};
```

## 3. 从 edit_decision.json 渲染

### 3.1 Python 调用

```python
import json
import subprocess
from pathlib import Path

def render_with_remotion(edit_decision_path: str, output_path: str):
    """调用 Remotion CLI 渲染视频"""
    # 1. 将 edit_decision 转为 Remotion props
    with open(edit_decision_path) as f:
        edit_decision = json.load(f)

    props = {
        "timeline": edit_decision["timeline"],
        "bgm": edit_decision.get("bgm", {}).get("path"),
    }

    props_path = "/tmp/remotion_props.json"
    with open(props_path, "w") as f:
        json.dump(props, f, ensure_ascii=False)

    # 2. 调用 Remotion CLI
    cmd = [
        "npx", "remotion", "render",
        "TravelVlog",           # composition id
        output_path,            # 输出路径
        "--props", props_path,  # 输入参数
        "--codec", "h264",
        "--crf", "18",
        "--pixel-format", "yuv420p",
        "--concurrency", "4",
    ]

    subprocess.run(cmd, check=True, cwd="/path/to/remotion/project")
```

## 4. Kinetic Typography（动态文字）

旅行 Vlog 常用的文字动画效果：

### 4.1 打字机效果

```tsx
const TypewriterText: React.FC<{text: string; fps: number}> = ({text, fps}) => {
  const frame = useCurrentFrame();
  const charsToShow = Math.floor(frame / 2);
  return <span>{text.slice(0, charsToShow)}</span>;
};
```

### 4.2 弹跳进入

```tsx
const BounceText: React.FC<{text: string}> = ({text}) => {
  const frame = useCurrentFrame();
  const y = spring({frame, fps: 30, config: {damping: 8, mass: 0.8}});
  return <div style={{transform: `translateY(${interpolate(y, [0, 1], [50, 0])}px)`}}>{text}</div>;
};
```

### 4.3 逐字动画

```tsx
const CharByChar: React.FC<{text: string}> = ({text}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{display: 'flex'}}>
      {text.split('').map((char, i) => {
        const delay = i * 3;
        const opacity = interpolate(frame - delay, [0, 5], [0, 1], {extrapolateLeft: 'clamp'});
        return <span key={i} style={{opacity}}>{char}</span>;
      })}
    </div>
  );
};
```

## 5. 渲染配置

### 5.1 抖音格式

```typescript
// remotion.config.ts
import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
Config.setCodec('h264');
Config.setCrf(18);
Config.setPixelFormat('yuv420p');
Config.setOutputLocation('out/video.mp4');
```

### 5.2 Composition 注册

```tsx
// src/Root.tsx
import {Composition} from 'remotion';
import {TravelVlog} from './TravelVlog';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="TravelVlog"
        component={TravelVlog}
        durationInFrames={900}  // 30s * 30fps
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{timeline: [], bgm: undefined}}
      />
    </>
  );
};
```

## 6. FFmpeg vs Remotion 选择

| 场景 | 推荐引擎 | 原因 |
|---|---|---|
| 纯素材剪辑 + 转场 | FFmpeg | 性能高，无需 Node 环境 |
| 需要动态文字动画 | Remotion | 文字动画控制力强 |
| Ken Burns + 字幕 | FFmpeg | 一条命令搞定 |
| Kinetic Typography 开头 | Remotion | 逐字动画效果好 |
| 批量渲染 | FFmpeg | 无 Node 启动开销 |
| 需要精确帧级控制 | Remotion | React 组件化管理 |

**推荐策略**：先用 FFmpeg 做主体剪辑，再用 Remotion 渲染开头动画片段，最后用 FFmpeg 拼接。
