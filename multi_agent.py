"""
多智能体系统
使用langgraph和supervisor模式实现
功能：
1. 图片问答（VL智能体）
2. 根据图片生成代码（Coder智能体）
3. 通用对话（Chat智能体）
"""
import asyncio
import base64
import io
from typing import Annotated, Literal, List, Dict
from typing_extensions import TypedDict
from collections import namedtuple

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import InjectedState
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from PIL import Image
from operator import add


import logging

logging.basicConfig(handlers=[logging.StreamHandler()], format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ==================== 本地tool ====================

from file_tools import create_folder, list_directory, delete_item, rename_item, move_file, write_file, read_file

# ==================== 模型配置 ====================
LLMInstance = namedtuple('LLMInstance', ['api_key', 'api_base', 'model_name'])



# 配置三个模型实例
chat_model_inst = LLMInstance(
    api_key='EMPTY',
    api_base='https://openai-api',
    model_name="gpt-5-nano"
)

vl_model_inst = LLMInstance(
    api_key='EMPTY',
    api_base='https://openai-api',
    model_name="gpt-4o-mini"  # 使用支持视觉的模型
)

coder_model_inst = LLMInstance(
    api_key='EMPTY',
    api_base='https://openai-api',
    model_name="gpt-4o-mini"  # 使用支持视觉和代码生成的模型
)



def get_llm(llm_instance: LLMInstance):
    """获取LLM实例"""
    if "qwen" in llm_instance.model_name:
        return ChatOpenAI(
            model=llm_instance.model_name,
            base_url=llm_instance.api_base,
            api_key=llm_instance.api_key,
            temperature=0.7,
            extra_body={"enable_thinking": False}
        )

    
    return ChatOpenAI(
        model=llm_instance.model_name,
        base_url=llm_instance.api_base,
        api_key=llm_instance.api_key,
        temperature=0.7
    )


# ==================== State定义 ====================
class ImgsMessages(TypedDict):
    img: Image.Image
    url: str = ""
    description: Dict = {}


class State(TypedDict):
    """
    状态定义
    messages: 消息列表
    imgs: 图片列表，每个图片包含img对象、url和描述
    """
    messages: MessagesState
    imgs: List[ImgsMessages]
    step: Annotated[int, add] = 0
    rollout: Annotated[List[str], add] = []
    current_task: str
    plan: str
    iteration_count: Annotated[int, add] = 0  # 迭代计数器
    last_agent: str = ""  # 记录上一个执行的智能体
    task: str = "" # 总任务


# ==================== MCP工具配置 ====================
mpc_client_config = {
    "math": {
        "command": "/data/conda_envs/shl_dev/bin/python",
        "args": ["/data/code/learn/mcp_learn/mcp_stdio.py"],
        "transport": "stdio"
    }
}


# ==================== 工具定义 ====================
async def get_mcp_tools():
    """获取MCP工具"""
    try:
        mcp_client = MultiServerMCPClient(mpc_client_config)
        tools = await mcp_client.get_tools()
        return tools
    except Exception as e:
        print(f"Warning: Failed to load MCP tools: {e}")
        return []
    
async def get_file_tools():
    """获取操作本地文件的工具"""
    return [create_folder, list_directory, delete_item, rename_item, move_file, write_file, read_file]


def encode_pil_image(image: Image.Image) -> str:
    """将PIL Image转换为base64字符串"""
    buffered = io.BytesIO()
    # 确保图片是RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def prepare_image_content(imgs: List[ImgsMessages]) -> List[Dict]:
    """准备图片内容，用于消息传递"""
    img_content = []
    for img_msg in imgs:
        if img_msg.get("img"):
            # 如果有PIL Image对象，转换为base64
            img_base64 = encode_pil_image(img_msg["img"])
            img_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        elif img_msg.get("url"):
            # 如果有URL，直接使用
            img_content.append({
                "type": "image_url",
                "image_url": {"url": img_msg["url"]}
            })
    return img_content


# ==================== 智能体切换工具 ====================
def decide_agent_with_llm(state: State) -> str:
    """使用LLM基于plan与历史消息决定应该调用哪个智能体。
    返回值限定为: "vl_agent" | "coder_agent" | "chat_agent"
    """
    llm = get_llm(chat_model_inst)
    messages = state.get("messages", [])
    rollout = state.get("rollout", [])
    plan = state.get("plan", "")
    imgs = state.get("imgs", [])

    has_images = len(imgs) > 0
    condensed_history = []
    try:
        # 仅取最近的若干轮以减小上下文
        tail = messages[-8:] if len(messages) > 8 else messages
        condensed_history = tail
    except Exception:
        condensed_history = messages

    system_prompt = (
        "你是路由调度器。根据对话历史、当前plan，选择最合适的智能体。\n"
        "- 如果需要基于图片理解/回答，选 vl_agent。\n"
        "- 如果需要产出或完善代码，选 coder_agent。\n"
        "- 其他通用对话与工具使用，选 chat_agent。\n"
        "只输出三个之一：vl_agent / coder_agent / chat_agent，不要输出其它内容。"
    )

    user_prompt = (
        f"current_task={state.get('current_task','')}\n"
        f"plan(可为空):\n{plan}\n"
        "请返回路由目标。"
    )

    try:
        resp = llm.invoke([SystemMessage(content=system_prompt)] + list(condensed_history) + [HumanMessage(content=user_prompt)])
        text = resp.content.strip().lower() if hasattr(resp, 'content') else str(resp).strip().lower()
        if "vl_agent" in text:
            return "vl_agent"
        if "coder_agent" in text or "code" in text:
            return "coder_agent"
        return "chat_agent"
    except Exception:
        # 兜底：无异常信息时根据是否有图片做最小化判定
        return "vl_agent" if has_images else "chat_agent"


# ==================== 智能体切换工具 ====================
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """创建智能体切换工具"""
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"
    
    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """切换到指定的智能体"""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        
        return Command(
            goto=agent_name,
            update={"messages": state["messages"] + [tool_message]},
        )
    return handoff_tool


def messages_change(messages, is_mmd=False):
    """
    将存在图片的消息体去掉，使文本大模型可使用messages
    
    Args:
        messages: 传递的消息
        is_mmd: 是否用于多模态模型
    """
    if is_mmd:
        return messages
    new_messages = []
    for msg in messages:
        # 兼容 LangChain Message 与 dict 两种格式
        content = None
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get('content')

        if isinstance(content, list):
            # 仅提取文本片段
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    new_messages.append({"role": "user", "content": item.get('text', '')})
        else:
            new_messages.append(msg)
    return new_messages


# ==================== Supervisor节点 ====================
def supervisor_node(state: State):
    """
    Supervisor节点：根据用户消息和图片内容路由到合适的智能体
    支持迭代循环，最多20次迭代
    """
    logger.info(f"rollout: {state.get('rollout', [])}")
    messages = state.get("messages", [])
    imgs = state.get("imgs", [])
    iteration_count = state.get("iteration_count", 0)
    last_agent = state.get("last_agent", "")
    
    # 检查迭代次数是否超过限制
    if iteration_count >= 20:
        logger.info(f"达到最大迭代次数 {iteration_count}，任务结束")
        return Command(
            goto=END,
            update={
                "rollout": [f"supervisor: 达到最大迭代次数 {iteration_count}，任务结束"],
                "iteration_count": 1
            }
        )
    
    last_message = messages_change(messages)[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    llm = get_llm(chat_model_inst)
    has_images = len(imgs) > 0

    # 首次：如有图片，必须先走VL进行结构分析；返回后在supervisor生成plan再继续
    if state.get("step") == 0 and has_images:
        return Command(
            goto="vl_agent",
            update={
                "rollout": ["supervisor: route -> vl_agent"], 
                "current_task": "图片问答", 
                "step": 1,
                "iteration_count": 1,
                "last_agent": "vl_agent"
            },
        )

    # 如果coder_agent刚完成，需要chat_agent校验
    if last_agent == "coder_agent":
        logger.info("coder_agent完成，需要chat_agent校验")
        # 提取coder_agent的所有输出，包括工具调用和结果
        # 从后往前遍历消息，收集所有相关的消息（assistant消息、tool调用、tool结果）
        coder_output_parts = []
        try:
            if messages:
                # 从后往前遍历，找到最后几条相关的消息
                # 通常模式是：assistant消息（可能包含tool_calls） -> tool消息（工具结果） -> assistant消息（最终输出）
                collected_indices = set()
                i = len(messages) - 1
                
                # 先找到最后一条assistant消息
                last_assistant_idx = -1
                while i >= 0:
                    msg = messages[i]
                    msg_role = msg.role if hasattr(msg, 'role') else (msg.get('role') if isinstance(msg, dict) else None)
                    if msg_role == 'assistant':
                        last_assistant_idx = i
                        break
                    i -= 1
                
                if last_assistant_idx >= 0:
                    # 从最后一条assistant消息开始，往前收集所有相关的消息
                    i = last_assistant_idx
                    while i >= 0 and i >= last_assistant_idx - 10:  # 最多往前找10条消息
                        if i in collected_indices:
                            i -= 1
                            continue
                            
                        msg = messages[i]
                        msg_role = msg.role if hasattr(msg, 'role') else (msg.get('role') if isinstance(msg, dict) else None)
                        
                        if msg_role == 'assistant':
                            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                            # 检查是否有工具调用
                            tool_calls = None
                            if hasattr(msg, 'tool_calls'):
                                tool_calls = msg.tool_calls
                            elif isinstance(msg, dict) and 'tool_calls' in msg:
                                tool_calls = msg['tool_calls']
                            
                            if tool_calls:
                                # 有工具调用
                                if content:
                                    coder_output_parts.insert(0, f"[输出] {content}")
                                for tool_call in tool_calls:
                                    if isinstance(tool_call, dict):
                                        fn = tool_call.get('function', {})
                                        if isinstance(fn, dict):
                                            fn_name = fn.get('name', '')
                                            fn_args = fn.get('arguments', '')
                                            if isinstance(fn_args, str):
                                                try:
                                                    import json
                                                    fn_args = json.loads(fn_args)
                                                except:
                                                    pass
                                            coder_output_parts.insert(0, f"[工具调用] {fn_name}({fn_args})")
                            elif content:
                                # 普通assistant消息
                                coder_output_parts.insert(0, f"[输出] {content}")
                            collected_indices.add(i)
                            i -= 1
                            
                        elif msg_role in ['tool', 'function']:
                            # 工具调用结果
                            tool_content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                            tool_name = msg.name if hasattr(msg, 'name') else msg.get('name', '')
                            coder_output_parts.insert(0, f"[工具结果] {tool_name}: {tool_content}")
                            collected_indices.add(i)
                            i -= 1
                        else:
                            # 遇到其他类型的消息，停止收集
                            break
                    
                    coder_output = "\n".join(coder_output_parts)
                else:
                    # 没找到assistant消息，至少取最后一条消息
                    if messages:
                        last_msg = messages[-1]
                        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                        coder_output = content
                    else:
                        coder_output = ""
        except Exception as e:
            logger.warning(f"提取coder_agent输出时出错: {e}")
            coder_output = ""
            # 兜底：至少取最后一条消息
            try:
                if messages:
                    last_msg = messages[-1]
                    coder_output = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            except Exception:
                pass
        
        # 创建校验提示，强调需要检查生成的文件
        validation_prompt = f"""请校验coder_agent生成的代码是否符合要求。

重要提示：
1. coder_agent可能已将代码写入文件，请使用read_file工具读取相关文件进行校验
2. 检查coder_agent的消息中提到的文件路径，读取这些文件的内容
3. 如果消息中提到了文件路径（如".html", ".css", ".js", ".py"等），请使用read_file工具读取这些文件

原始需求/计划：
{state.get('plan', '无计划')}

Coder Agent的输出消息：
{coder_output}

请执行以下步骤进行校验：
1. 首先检查coder_agent的消息中是否提到了文件路径
2. 如果提到了文件路径，使用read_file工具读取这些文件的内容
3. 评估代码是否完整实现了需求
4. 评估代码质量是否合格（语法、逻辑、结构等）
5. 检查是否需要进一步修改或完善

如果代码已经完成且质量合格，请回复"任务已完成"。
如果需要修改，请指出具体问题和改进建议，包括需要修改的文件路径和具体修改内容。"""
        
        try:
            validation_msg = HumanMessage(content=validation_prompt)
            update_payload = {
                "step": 1, 
                "rollout": [f"supervisor: coder_agent完成，路由到chat_agent进行校验 (迭代 {iteration_count + 1})"], 
                "current_task": "代码校验",
                "iteration_count": 1,
                "last_agent": "chat_agent"
            }
            try:
                update_payload["messages"] = list(messages) + [validation_msg]
            except Exception:
                update_payload["messages"] = [*messages, validation_msg]
            return Command(
                goto="chat_agent",
                update=update_payload,
            )
        except Exception as e:
            logger.error(f"创建校验消息失败: {e}")

    # 检查任务是否完成（通过LLM判断）
    if iteration_count > 0:  # 至少完成一次迭代后才检查
        try:
            completion_check_prompt = [
                SystemMessage(content="你是任务完成判断器。根据对话历史和当前状态，判断任务是否已经完成。如果任务已完成，回复'任务已完成'；如果需要继续，回复'需要继续'。"),
                HumanMessage(content=f"""当前迭代次数: {iteration_count}/20
总任务： {state.get("task", "")}
计划: {state.get('plan', '无计划')}

请判断任务是否已经完成。""")
            ]
            # 添加最近的消息历史
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            completion_response = llm.invoke(completion_check_prompt + list(recent_messages))
            completion_text = completion_response.content.strip().lower() if hasattr(completion_response, 'content') else str(completion_response).strip().lower()
            
            if "任务已完成" in completion_text or "已完成" in completion_text or "完成" in completion_text:
                logger.info(f"任务已完成，迭代次数: {iteration_count}")
                return Command(
                    goto=END,
                    update={
                        "rollout": [f"supervisor: 任务已完成 (迭代 {iteration_count} 次)"],
                        "iteration_count": 1
                    }
                )
        except Exception as e:
            logger.warning(f"任务完成检查失败: {e}，继续执行")

    # 非首次路由：先在有图且无plan的情况下生成plan，然后调用LLM判定
    plan = state.get("plan") or ""
    plan_added_message = None
    if has_images and not plan:
        try:
            # 尝试融合VL输出的描述
            vl_desc = "\n".join([
                (im.get("description", {}) or {}).get("vl_output", "") for im in (imgs or [])
            ])
            plan_prompt = [
                SystemMessage(content="你是资深软件架构师。请基于对话与（可能的）图片结构描述，产出实现计划（Markdown）。要求：目标、关键模块、数据结构/接口、步骤拆分、验收标准。"),
                HumanMessage(content=(str(content) + ("\n\n[图片结构]" + vl_desc if vl_desc else "")))
            ]
            plan_msg = llm.invoke(plan_prompt)
            plan = plan_msg.content if hasattr(plan_msg, 'content') else str(plan_msg)
        except Exception:
            plan = "## 实现计划\n- 待补充：计划生成失败，稍后由Coder完善"
        # 需要更新state的plan，并将plan注入messages
        state.update({"plan": plan})
        try:
            plan_added_message = AIMessage(content=plan)
        except Exception:
            plan_added_message = {"role": "assistant", "content": plan}
    
    # 使用LLM决定下一个智能体
    target = decide_agent_with_llm(state)
    next_task = {
        "vl_agent": "图片问答",
        "coder_agent": "代码生成",
        "chat_agent": "通用对话",
    }.get(target, "通用对话")

    update_payload = {
        "step": 1, 
        "rollout": [f"supervisor: route -> {target} (迭代 {iteration_count + 1})"], 
        "current_task": next_task, 
        "plan": plan,
        "iteration_count": 1,
        "last_agent": target
    }
    if plan_added_message is not None:
        try:
            update_payload["messages"] = list(messages) + [plan_added_message]
        except Exception:
            update_payload["messages"] = [*messages, plan_added_message]
    return Command(
        goto=target,
        update=update_payload,
    )


# ==================== VL智能体（图片问答） ====================
# 缓存智能体实例，避免重复创建
_vl_agent_cache = None
_coder_agent_cache = None
_chat_agent_cache = None


async def create_vl_agent():
    """创建VL智能体，用于图片问答"""
    global _vl_agent_cache
    if _vl_agent_cache is not None:
        return _vl_agent_cache
    
    llm = get_llm(vl_model_inst)
    
    # 获取MCP工具（不再在此创建handoff工具，由supervisor统一路由）
    mcp_tools = await get_file_tools()
    tools = mcp_tools
    
    prompt = """你是一个专业的视觉问答智能体，擅长分析和回答关于图片的问题。
    当用户上传图片并提问时，你需要：
    1. 仔细分析图片内容
    2. 理解用户的问题
    3. 提供准确、详细的回答
    4. 识别UI结构/布局/文本/颜色
    
    路由和智能体切换由上层的supervisor负责，你只需专注于图片理解与回答。
    """
    
    _vl_agent_cache = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        name="vl_agent"
    )
    return _vl_agent_cache


def vl_agent_wrapper(state: State):
    """VL智能体包装器，处理图片信息"""
    logger.info(f"rollout: {state.get('rollout', [])}")
    messages = state.get("messages", [])
    imgs = state.get("imgs", [])
    
    # 创建VL智能体并调用
    agent = asyncio.run(create_vl_agent())
    result = agent.invoke({"messages": messages})
    # 尝试提取VL输出内容并写入到每张图片的description
    vl_content = None
    try:
        if isinstance(result, list) and result:
            last_msg = result[-1]
            vl_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        elif isinstance(result, dict) and result.get("messages"):
            rmsgs = result.get("messages")
            if isinstance(rmsgs, list) and rmsgs:
                last_msg = rmsgs[-1]
                vl_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
    except Exception:
        vl_content = None
    if vl_content and imgs:
        try:
            updated_imgs = []
            for im in imgs:
                new_im = dict(im)
                new_im["description"] = {"vl_output": vl_content}
                updated_imgs.append(new_im)
        except Exception:
            updated_imgs = imgs
    else:
        updated_imgs = imgs
    # 更新step与rollout（聚合追加），标记last_agent为vl_agent
    if isinstance(result, dict):
        result.update({
            "step": 1, 
            "rollout": ["vl_agent"], 
            "imgs": updated_imgs,
            "last_agent": "vl_agent"
        })
        return result
    return {
        "messages": result if isinstance(result, list) else [result], 
        "step": 1, 
        "rollout": ["vl_agent"], 
        "imgs": updated_imgs,
        "last_agent": "vl_agent"
    }


# ==================== Coder智能体（代码生成） ====================
async def create_coder_agent():
    """创建Coder智能体，用于根据图片生成代码"""
    global _coder_agent_cache
    if _coder_agent_cache is not None:
        return _coder_agent_cache
    
    llm = get_llm(coder_model_inst)
    
    # 获取本地文件操作工具（路由交由supervisor，移除handoff工具）
    mcp_tools = await get_file_tools()
    tools = mcp_tools
    
    prompt = """你是一个专业的代码生成智能体，擅长根据图片生成代码。
    当用户上传图片并请求生成代码时，你需要：
    1. 仔细分析图片中的代码、界面或需求
    2. 理解用户的具体需求
    3. 生成高质量、可运行的代码
    4. 提供详细的代码说明和注释
    
    路由和智能体切换由上层的supervisor负责，你只需专注于代码生成。
    """
    
    _coder_agent_cache = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        name="coder_agent"
    )
    return _coder_agent_cache


def coder_agent_wrapper(state: State):
    """Coder智能体包装器，处理图片信息"""
    logger.info(f"rollout: {state.get('rollout', [])}")
    messages = state.get("messages", [])
    imgs = state.get("imgs", [])
    
    # 创建Coder智能体并调用
    agent = asyncio.run(create_coder_agent())
    result = agent.invoke({"messages": messages})
    # 更新step与rollout（聚合追加），标记last_agent为coder_agent
    if isinstance(result, dict):
        result.update({
            "step": 1, 
            "rollout": ["coder_agent"],
            "last_agent": "coder_agent"
        })
        return result
    return {
        "messages": result if isinstance(result, list) else [result], 
        "step": 1, 
        "rollout": ["coder_agent"],
        "last_agent": "coder_agent"
    }


# ==================== Chat智能体（通用对话） ====================
async def create_chat_agent():
    """创建Chat智能体，用于通用对话和代码校验"""
    global _chat_agent_cache
    if _chat_agent_cache is not None:
        return _chat_agent_cache
    
    llm = get_llm(chat_model_inst)
    
    # 获取MCP工具和文件工具（特别是read_file用于校验代码文件）
    mcp_tools = await get_mcp_tools()
    file_tools = await get_file_tools()
    tools = mcp_tools + file_tools
    
    prompt = """你是一个友好的通用对话智能体，擅长回答各种问题和使用工具。
    你还可以进行代码校验工作，当需要校验代码时，请使用read_file工具读取相关文件，然后评估代码质量。
    路由和智能体切换由上层的supervisor负责，你只需专注于高质量对话、工具使用和代码校验。
    """
    
    _chat_agent_cache = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        name="chat_agent"
    )
    return _chat_agent_cache


def chat_agent_wrapper(state: State):
    """Chat智能体包装器"""
    logger.info(f"rollout: {state.get('rollout', [])}")
    messages = state.get("messages", [])
    
    # 创建Chat智能体并调用
    agent = asyncio.run(create_chat_agent())
    result = agent.invoke({"messages": messages})
    # 更新step与rollout（聚合追加），标记last_agent为chat_agent
    if isinstance(result, dict):
        result.update({
            "step": 1, 
            "rollout": ["chat_agent: completed"],
            "last_agent": "chat_agent"
        })
        return result
    return {
        "messages": result if isinstance(result, list) else [result], 
        "step": 1, 
        "rollout": ["chat_agent: completed"],
        "last_agent": "chat_agent"
    }


# ==================== 构建图 ====================
def create_multi_agent_graph():
    """创建多智能体图，支持迭代循环"""
    graph = StateGraph(State)
    
    # 添加节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("vl_agent", vl_agent_wrapper)
    graph.add_node("coder_agent", coder_agent_wrapper)
    graph.add_node("chat_agent", chat_agent_wrapper)
    
    # 添加边
    graph.add_edge(START, "supervisor")
    # 所有智能体完成后都回到supervisor，由supervisor决定是否继续或结束
    graph.add_edge("vl_agent", "supervisor")
    graph.add_edge("coder_agent", "supervisor")
    graph.add_edge("chat_agent", "supervisor")
    
    return graph.compile()


# ==================== 主函数 ====================
if __name__ == "__main__":
    import base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    # 创建图
    graph = create_multi_agent_graph()
    
    # 测试用例1：图片问答
    print("=" * 50)
    print("测试用例1：图片问答")
    print("=" * 50)
    
    # 注意：这里需要实际的图片路径或URL
    
    text_prompt1 = """分析这张图片的结构，为代码实现提供具体的描述"""
    text_prompt2 = """使用前端代码实现这个图片"""
    text_prompt3 = """你说1+1等于多少呢？"""
    tetx_prompt4 = """这是两个设计图，功能是网页进行录音；要求按照这个录音设计图，设计一个可录音的页面, 要求页面可真是录音;要求页面可交互;同时使用code要实现设计好页面，并输出到文件。"""
    
    
    img_path = "/data/code/learn/image_stack_find/resized_image.png"
    img_path = "/data/code/learn/work/agent_vl/代码生成测试_录音页面.png"
    
    text_prompt = tetx_prompt4
    messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encode_image(img_path)}"},
                            },
                            {"type": "text", "text": text_prompt},
                        ],
                    }
                ]
    
    # class State(TypedDict):
    #     """
    #     状态定义
    #     messages: 消息列表
    #     imgs: 图片列表，每个图片包含img对象、url和描述
    #     """
    #     messages: MessagesState
    #     imgs: List[ImgsMessages]
    #     step: int
    #     rollout: List[str]
    
    # 运行测试（取消注释以运行）
    result = graph.invoke({
        "messages": messages, 
        "step": 0, 
        "rollout": ['start'], 
        'imgs': [{"img": Image.open(img_path)}],
        "iteration_count": 0,
        "last_agent": "",
        "current_task": "",
        "plan": "",
        "task": text_prompt  # 用户输入的最后一个
    })
    # print(result)
