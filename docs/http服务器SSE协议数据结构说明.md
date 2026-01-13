# 深度研究应用 SSE Chunk 说明

本应用通过 **Server-Sent Events (SSE)** 实时向使用者推送研究过程中的各类事件（chunk）。
使用者可以**实时感知研究的思考过程、子调研任务的创建与执行，以及最终报告的生成**。

所有 SSE 消息均采用如下格式：

```
event: <event_name>
data: <json>
```

---

## 一、Chunk 总览

SSE 过程中，使用者**可能会看到以下 5 种 chunk**（按研究流程大致顺序）：

| event 名称              | 说明                  |
| --------------------- | ------------------- |
| `ai_message_chunk`    | 主研究智能体的实时思考 / 说明性输出 |
| `new_sub_research`    | 创建一个新的子调研任务         |
| `sub_research_action` | 子调研任务正在执行具体动作       |
| `sub_research_done`   | 某个子调研任务完成           |
| `report_chunk`        | 最终研究报告的流式输出         |

---

## 二、各类 Chunk 详细说明

### 1️⃣ `ai_message_chunk` —— 主智能体的实时输出

**用途**
用于向使用者展示**主研究智能体的思考、解释、过渡性说明或中间结论**。

**典型场景**

* 解释研究目标
* 说明接下来要拆分哪些子问题
* 总结当前阶段的研究发现
* 在报告生成前做整体说明

**数据结构**

```json
{
  "content": "string",
  "end": false
}
```

**字段说明**

* `content`：当前输出的文本内容（流式）
* `end`：

  * `false`：该消息还会继续输出
  * `true`：该段 AI 输出结束

**使用者体验**

> 像 ChatGPT 一样逐字/逐段看到 AI 的实时输出。

---

### 2️⃣ `new_sub_research` —— 创建新的子调研任务

**用途**
当主研究智能体决定**派遣一个子智能体进行专项研究**时，发送该事件。

**子调研类型（目前支持）**

* `网络调研`
* `知识库调研`

**数据结构**

```json
{
  "id": "string",
  "type": "网络调研",
  "title": "string",
  "description": "string"
}
```

**字段说明**

* `id`：子调研任务唯一 ID（用于后续事件关联）
* `type`：调研类型

  * `"网络调研"`
  * `"知识库调研"`
* `title`：该子调研的简要标题
* `description`：该子调研要解决的问题说明

**使用者体验**

> 明确看到：
> 「系统正在启动一个新的调研任务：XXX」

---

### 3️⃣ `sub_research_action` —— 子调研的具体执行动作

**用途**
展示子调研智能体在执行过程中的**具体行动步骤**。

**典型动作**

* 网络搜索某个关键词
* 查询某个知识库
* 聚合/筛选搜索结果

**数据结构**

```json
{
  "id": "string",
  "type": "网络搜索",
  "description": "string"
}
```

**字段说明**

* `id`：对应的子调研任务 ID
* `type`：

  * `"网络搜索"`
  * `"知识库搜索"`
* `description`：本次行动的具体说明

**使用者体验**

> 让研究过程“可见”：
> 「正在搜索：XXX」
> 「正在查询知识库：XXX」

---

### 4️⃣ `sub_research_done` —— 子调研完成

**用途**
标记某个子调研任务**已完成，结果已返回给主智能体**。

**数据结构**

```json
{
  "id": "string"
}
```

**字段说明**

* `id`：完成的子调研任务 ID

**使用者体验**

> 明确感知进度：
> 「子调研任务 XXX 已完成」

---

### 5️⃣ `report_chunk` —— 最终研究报告的流式输出

**用途**
当所有子调研完成后，主研究智能体开始**整合结果并生成最终研究报告**。

**数据结构**

```json
{
  "content": "string",
  "end": false
}
```

**字段说明**

* `content`：报告正文内容（流式）
* `end`：

  * `false`：报告尚未生成完
  * `true`：报告生成结束（SSE 可视为结束）

**使用者体验**

> 像阅读一篇实时生成的研究报告
> （而不是等所有内容一次性返回）

---

## 三、完整典型流程示意

使用者看到的 SSE 流可能类似：

1. `ai_message_chunk`
   → 说明研究背景和目标
2. `new_sub_research`
   → 创建「网络调研：XXX」
3. `sub_research_action`
   → 网络搜索 A
4. `sub_research_action`
   → 网络搜索 B
5. `sub_research_done`
   → 网络调研完成
6. `new_sub_research`
   → 创建「知识库调研：YYY」
7. `sub_research_action`
   → 知识库搜索
8. `sub_research_done`
9. `ai_message_chunk`
   → 汇总调研结论
10. `report_chunk`（多次）
11. `report_chunk (end=true)`
    → 报告完成
