# 钩子公式库（动态版）

> 本文件由 `tools/trends/douyin_trend_analyzer.py` 每周自动更新。
> Agent 在 `script_generation` 阶段从这里选取钩子公式。

## 当前热门钩子公式

> ⚠️ 以下内容为初始模板，系统运行后会被趋势分析模块自动覆盖。

### Tier 1 — 高完播率（>60%）

```yaml
- hook_type: suspense
  template: "去了{N}次{destination}才知道的{M}件事"
  example: "去了5次富士山才知道的3个隐藏机位"
  effectiveness: 0.68
  best_for: [攻略, 隐藏景点]

- hook_type: contrast
  template: "以为{destination}很{negative}，结果..."
  example: "以为京都很无聊，结果不想走了"
  effectiveness: 0.63
  best_for: [小众目的地, 反差感]

- hook_type: shock
  template: "[无文字，纯画面冲击3秒]"
  example: "[富士山日出金色时刻航拍]"
  effectiveness: 0.72
  best_for: [风景大片, 航拍素材]
```

### Tier 2 — 中完播率（45-60%）

```yaml
- hook_type: number
  template: "{N}天{M}元玩转{destination}"
  example: "3天800元玩转东京"
  effectiveness: 0.55
  best_for: [穷游, 攻略]

- hook_type: resonance
  template: "如果只剩一天在{destination}..."
  example: "如果只剩一天在京都..."
  effectiveness: 0.52
  best_for: [情感向, 治愈系]

- hook_type: question
  template: "为什么{destination}人都去{place}？"
  example: "为什么日本人都去镰仓？"
  effectiveness: 0.48
  best_for: [文化探秘, 深度游]
```

### Tier 3 — 已过时（禁止使用）

```yaml
- hook_type: deprecated
  templates:
    - "今天带大家去..."
    - "哈喽大家好我是..."
    - "你们知道吗？..."
    - "不看后悔的..."
  effectiveness: 0.25
  note: "2023年前的钩子模式，完播率已降至25%以下，禁止使用"
```

## 钩子选择决策树

```
1. 素材中有航拍/大景画面吗？
   ├─ 是 → 优先使用 shock 型（纯画面冲击）
   └─ 否 → 继续

2. 目的地是热门还是小众？
   ├─ 热门 → 优先 suspense 型（"去了N次才知道"）
   └─ 小众 → 优先 contrast 型（"以为很无聊"）

3. 有美食/人文素材吗？
   ├─ 是 → 可以用 resonance 型
   └─ 否 → 不用 resonance 型

4. 视频时长 < 20s？
   ├─ 是 → 必须用 shock 型（没时间说钩子文字）
   └─ 否 → 可用任意类型
```

## 更新日志

| 日期 | 更新内容 | 数据来源 |
|---|---|---|
| 2026-08-11 | 初始模板创建 | 手动编写 |

> 下次趋势分析运行后，此文件将被自动更新。
