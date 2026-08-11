# FFmpeg 技术参考

> Agent 在 `compose` 阶段调用 FFmpeg 时的命令参考。
> 本文件不是教程，是可查可用的命令模板。

## 1. 基础转码

### 1.1 标准化为 H.264 1080p 30fps

```bash
ffmpeg -i input.mov \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
  -r 30 -c:v libx264 -preset medium -crf 18 \
  -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart \
  output.mp4
```

### 1.2 转为抖音竖屏 9:16

```bash
ffmpeg -i input.mp4 \
  -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -preset medium -crf 18 -r 30 \
  -c:a aac -b:a 192k \
  output_9x16.mp4
```

### 1.3 批量转码脚本

```python
import subprocess
from pathlib import Path

def transcode_to_standard(input_path: str, output_path: str, target_w=1920, target_h=1080, fps=30):
    """转码为标准格式"""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
               f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        "-y", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
```

## 2. 剪辑操作

### 2.1 截取片段

```bash
# 按时间截取（精确）
ffmpeg -ss 00:00:10.5 -i input.mp4 -t 5.0 -c copy output_clip.mp4

# 按秒截取
ffmpeg -ss 10.5 -i input.mp4 -t 5.0 -c copy output_clip.mp4
```

### 2.2 拼接多个片段

```bash
# 方法1: concat demuxer（推荐，无损）
# 先创建 filelist.txt:
# file 'clip1.mp4'
# file 'clip2.mp4'
# file 'clip3.mp4'
ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4

# 方法2: concat filter（需要统一编码格式时）
ffmpeg -i clip1.mp4 -i clip2.mp4 -i clip3.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" output.mp4
```

### 2.3 裁剪画面

```bash
# 居中裁剪为 9:16
ffmpeg -i input.mp4 \
  -vf "crop=ih*9/16:ih,scale=1080:1920" \
  output_cropped.mp4
```

## 3. 转场效果

### 3.1 淡入淡出

```bash
# 视频开头 1 秒淡入，结尾 1 秒淡出
ffmpeg -i input.mp4 \
  -vf "fade=t=in:st=0:d=1,fade=t=out:st=29:d=1" \
  -af "afade=t=in:st=0:d=0.5,afade=t=out:st=29.5:d=0.5" \
  output_fade.mp4
```

### 3.2 叠化（crossfade）

```bash
# 两个视频之间 0.5 秒叠化
ffmpeg -i clip1.mp4 -i clip2.mp4 \
  -filter_complex \
  "[0:v]trim=0:5,setpts=PTS-STARTPTS[v0];\
   [1:v]trim=0:5,setpts=PTS-STARTPTS[v1];\
   [v0][v1]xfade=transition=fade:duration=0.5:offset=4.5[v]" \
  -map "[v]" output_xfade.mp4
```

### 3.3 xfade 可用转场类型

```
fade, wipeleft, wiperight, wipeup, wipedown,
slideleft, slideright, slideup, slidedown,
circlecrop, rectcrop, distance, smoothleft, smoothright, smoothup, smoothdown,
circleopen, circleclose, vertopen, vertclose, horzopen, horzclose,
dissolve, pixelize, diagtl, diagtr, diagbl, diagbr, hlslice, hrslice, vuslice, vdslice
```

## 4. 字幕处理

### 4.1 硬字幕（烧录到画面）

```bash
# 使用 SRT 字幕文件
ffmpeg -i input.mp4 \
  -vf "subtitles=subtitles.srt:force_style='FontSize=24,FontName=Noto Sans CJK SC,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2'" \
  output_subtitled.mp4
```

### 4.2 动态字幕（drawtext）

```bash
# 在画面底部添加文字
ffmpeg -i input.mp4 \
  -vf "drawtext=text='富士山日出':fontfile=/path/to/font.ttf:fontsize=48:fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-100:enable='between(t,2,5)'" \
  output_text.mp4
```

### 4.3 ASS 字幕（高级样式）

```bash
ffmpeg -i input.mp4 -vf "ass=subtitle.ass" output.mp4
```

## 5. 音频处理

### 5.1 音量归一化

```bash
# 两遍 loudnorm（推荐）
ffmpeg -i input.mp4 -af "loudnorm=I=-14:TP=-1:LRA=11" -c:v copy output_normalized.mp4
```

### 5.2 混合 BGM 和旁白

```bash
ffmpeg -i video.mp4 -i bgm.mp3 -i voiceover.mp3 \
  -filter_complex "[1:a]volume=0.4[bg];[2:a]volume=1.0[vo];[bg][vo]amix=inputs=2:duration=longest[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac output_mixed.mp4
```

### 5.3 音频淡入淡出

```bash
ffmpeg -i input.mp4 -af "afade=t=in:st=0:d=0.5,afade=t=out:st=29.5:d=0.5" output.mp4
```

## 6. 质检命令

### 6.1 获取视频信息

```bash
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4
```

### 6.2 检测黑帧

```bash
ffmpeg -i input.mp4 -vf "blackdetect=d=0.5:pix_th=0.10" -an -f null - 2>&1 | grep blackdetect
```

### 6.3 检测静音段

```bash
ffmpeg -i input.mp4 -af "silencedetect=noise=-60dB:d=1" -f null - 2>&1 | grep silence
```

### 6.4 音量分析

```bash
ffmpeg -i input.mp4 -af "volumedetect" -f null - 2>&1 | grep -E "mean_volume|max_volume"
```

### 6.5 场景检测（用于幻灯片风险检测）

```bash
ffmpeg -i input.mp4 -vf "select='gt(scene,0.02)',showinfo" -f null - 2>&1 | grep scene
```

## 7. Ken Burns 效果

```bash
# 缓慢缩放
ffmpeg -i input.mp4 \
  -vf "scale=8000:-1,zoompan=z='min(zoom+0.0005,1.5)':d=125:s=1080x1920:fps=30" \
  output_kenburns.mp4

# 缓慢平移
ffmpeg -i input.mp4 \
  -vf "scale=8000:-1,zoompan=x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':z=1.3:d=125:s=1080x1920:fps=30" \
  output_pan.mp4
```

## 8. 完整渲染流程

```python
def render_video(edit_decision: dict, output_path: str):
    """从 edit_decision 渲染最终视频"""
    clips = edit_decision["timeline"]

    # Step 1: 截取每个片段
    clip_files = []
    for i, clip in enumerate(clips):
        clip_file = f"/tmp/clip_{i}.mp4"
        extract_clip(clip["source_path"], clip["in_sec"], clip["out_sec"], clip_file)
        clip_files.append(clip_file)

    # Step 2: 拼接
    concat_file = "/tmp/concat.mp4"
    concat_clips(clip_files, concat_file)

    # Step 3: 添加转场
    transitioned = "/tmp/transitioned.mp4"
    apply_transitions(concat_file, clips, transitioned)

    # Step 4: 添加字幕
    subtitled = "/tmp/subtitled.mp4"
    add_subtitles(transitioned, clips, subtitled)

    # Step 5: 混合音频
    if edit_decision.get("bgm"):
        mixed = "/tmp/mixed.mp4"
        mix_audio(subtitled, edit_decision["bgm"], edit_decision.get("voiceover"), mixed)
    else:
        mixed = subtitled

    # Step 6: 输出为抖音格式
    transcode_to_douyin(mixed, output_path)
```
