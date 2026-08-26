# 钩子公式库（动态版）
> 由 douyin_trend_analyzer 于 2026-08-16 18:36 自动更新
> Agent 在 `script_generation` 阶段从这里选取钩子公式。

## 当前热门钩子公式

### Tier 1 — 高完播率（>60%）

```yaml
- hook_type: shock
  template: "[无文字，纯画面冲击3秒]"
  example: "[富士山日出金色时刻航拍]"
  effectiveness: 0.72
  tier: Tier 1

- hook_type: suspense
  template: "去了{N}次{destination}才知道的{M}件事"
  example: "去了5次富士山才知道的3个隐藏机位"
  effectiveness: 0.68
  tier: Tier 1

- hook_type: resonance
  template: "你有没有过，看到{destination}的风景就想哭的瞬间"
  example: "你有没有过，看到新疆的风景就想哭的瞬间"
  effectiveness: 0.66
  tier: Tier 1

- hook_type: contrast
  template: "以为{destination}很{negative}，结果..."
  example: "以为京都很无聊，结果不想走了"
  effectiveness: 0.63
  tier: Tier 1

- hook_type: question
  template: "为什么{destination}的{element}能让人沉默"
  example: "为什么新疆的雪山能让人沉默"
  effectiveness: 0.62
  tier: Tier 1

- hook_type: first_person
  template: "我在{destination}遇到了{M}件不可思议的事"
  example: "我在新疆遇到了3件不可思议的事"
  effectiveness: 0.61
  tier: Tier 1

- hook_type: emotional
  template: "如果你也累了，就去{destination}看看"
  example: "如果你也累了，就去新疆看看"
  effectiveness: 0.60
  tier: Tier 1
```

### Tier 2 — 中等完播率（40%-60%）

```yaml
- hook_type: number
  template: "{N}天{M}元玩转{destination}"
  example: "3天800元玩转东京"
  effectiveness: 0.55
  tier: Tier 2

- hook_type: resonance_time
  template: "如果只剩一天在{destination}..."
  example: "如果只剩一天在京都..."
  effectiveness: 0.52
  tier: Tier 2

- hook_type: before_after
  template: "去{destination}之前vs之后，判若两人"
  example: "去新疆之前vs之后，判若两人"
  effectiveness: 0.50
  tier: Tier 2

- hook_type: secret
  template: "{destination}有个地方，本地人都不想告诉你"
  example: "新疆有个地方，本地人都不想告诉你"
  effectiveness: 0.49
  tier: Tier 2

- hook_type: sensory
  template: "在{destination}，{sense}到了从未有过的感觉"
  example: "在新疆，闻到了从未有过的自由"
  effectiveness: 0.48
  tier: Tier 2

- hook_type: contrast_age
  template: "{N}岁第一次去{destination}，被骂醒了"
  example: "30岁第一次去新疆，被骂醒了"
  effectiveness: 0.47
  tier: Tier 2

- hook_type: regret
  template: "后悔没早点去{destination}的{M}个地方"
  example: "后悔没早点去新疆的3个地方"
  effectiveness: 0.46
  tier: Tier 2

- hook_type: alone
  template: "一个人去{destination}，{N}天后不想回来"
  example: "一个人去新疆，7天后不想回来"
  effectiveness: 0.45
  tier: Tier 2

- hook_type: misconception
  template: "所有人都说{destination}不值得，直到我去了"
  example: "所有人都说新疆不值得，直到我去了"
  effectiveness: 0.44
  tier: Tier 2

- hook_type: milestone
  template: "走完{destination}这{M}条路，才算没白活"
  example: "走完新疆这3条路，才算没白活"
  effectiveness: 0.43
  tier: Tier 2

- hook_type: comparison
  template: "{destination} vs {destination2}，差距不止一点"
  example: "新疆 vs 西藏，差距不止一点"
  effectiveness: 0.42
  tier: Tier 2

- hook_type: season
  template: "在{season}的{destination}，我看到了另一个世界"
  example: "在秋天的喀纳斯，我看到了另一个世界"
  effectiveness: 0.41
  tier: Tier 2

- hook_type: night
  template: "{destination}的夜晚，藏着白天看不见的东西"
  example: "新疆的夜晚，藏着白天看不见的东西"
  effectiveness: 0.40
  tier: Tier 2
```

## 更新日志

| 日期 | 更新内容 | 数据来源 |
|---|---|---|
| 2026-08-16 | 自动更新（5个钩子公式） | 抖音趋势分析 |
| 2026-08-22 | 手动扩充至20条，新增情感共鸣/悬念/反差类 | 人工补充 |
