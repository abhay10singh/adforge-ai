import os
import time
import json
import io
import base64
import requests
from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
from PIL import Image, ImageOps
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
import cloudinary
import cloudinary.uploader
from openai import OpenAI
from langchain_core.tools import tool
from rembg import remove
from io import BytesIO
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
openai_client = OpenAI()
cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key = os.environ.get("CLOUDINARY_API_KEY"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")
)

class AdState(TypedDict):
    # =========================
    # INPUTS (User / System)
    # =========================
    product_image_path: str
    user_request: str
    audience: Optional[str]
    brand_tone: Optional[str]

    target_width: int
    target_height: int

    # =========================
    # CREATIVE / DIRECTOR OUTPUT
    # =========================
    style: str

    background_prompt: str
    negative_prompts: Optional[str]
    product_Situation_Description : str
    headline: str
    headline_position: str              # ADDED
    subcopy: Optional[str]
    cta_text: str
    cta_position: str                   # ADDED

    # Typography
    headline_font_primary: str
    headline_font_secondary: str
    headline_size_pct: float
    subcopy_size_pct: Optional[float]
    cta_font: str
    cta_size_pct: float

    # Colors
    headline_color: str
    text_color: Optional[str]
    palette: List[str]

    # CTA Styling
    cta_button_style: Dict[str, Any]

    # =========================
    # PRODUCT GEOMETRY (CRITICAL)
    # =========================
    product_scale: float
    product_position: Dict[str, float]
    product_rotation: float

    product_box_hint: Optional[Dict[str, float]]
    safe_text_margin_pct: float

    # =========================
    # VISUAL DIRECTIVES (ADDED)
    # =========================
    props: List[str]
    lighting_directive: Optional[str]
    texture_or_treatment: Optional[str]
    contrast_guidance: Optional[str]

    # =========================
    # IMAGE FILES (LOCAL)
    # =========================
    source_filename: str
    mask_filename: str
    product_layer_filename: Optional[str]

    final_image_path: Optional[str]
    background_image_path: Optional[str]

    # =========================
    # CLOUD / URL OUTPUTS
    # =========================
    image_url_for_background: Optional[str]
    image_url_for_AD: Optional[str]
    cloudinary_id: Optional[str]

    # =========================
    # VISION / QUALITY LOOP
    # =========================
    quality_score: int
    retry_count: int
    feedback_logs: List[str]

    best_score: int
    best_image_url: Optional[str]

    # =========================
    # DEBUG / META
    # =========================
    export_notes: Optional[str]
    confidence: Optional[float]
    error: Optional[str]

class ImageReview(BaseModel):
    """Analysis of the Generated Advertisement."""
    
    ImageRating: int = Field(
        ..., 
        description="Rate 1-10. Be harsh. Any spelling mistake or unclear text must reduce the score below 10.",
        ge=1, 
        le=10
    )
    Potential_errors: List[str] = Field(
        ..., 
        description="List of direct, imperative fix instructions. Example: 'Fix spelling of Shop', 'Increase contrast'."
    )


client = OpenAI()

def clamp(v, lo, hi): 
    try: return max(lo, min(hi, float(v)))
    except: return lo
def parse_float(d, key, default): 
    try: return float(d[key])
    except: return default

        
def parsing_image_parameters(state : AdState):
    The_Prompt = f"""You are an expert Creative Director for high-conversion visual ads. Analyze the attached {int(state['target_width'])} x {int(state['target_height'])} product image with the user request : {state.get('user_request',"")}and produce a single, production-ready ad concept tailored to this specific product: composition, background, lighting, props, typography, colors, and exact geometry values that downstream nodes will apply.

    REQUIREMENTS:
    1. Return ONLY valid JSON and nothing else.
    2. Return a single object named "candidate" matching the schema below.
    3. Enforce numeric ranges: product_scale 0.20-0.90, x_pct/y_pct 0.00-1.00, rotation -45..45, headline_size_pct 0.02-0.15, subcopy_size_pct 0.012-0.06, cta_size_pct 0.018-0.08, safe_text_margin_pct 0.0-0.20, confidence 0.0-1.0.
    4. Headline ≤ 6 words. CTA ≤ 3 words. Subcopy ≤ 10 words.
    5. Provide fonts as common family names (two choices). Provide hex colors.
    6. Background_prompt must be a concise, image-edit-ready sentence (no text instructions like "add headline").
    7. product_Situation_Description is essentially the situation where to product is there either stand along or along with a Model (human) using it or wearing (depending upon the product) but product should remain same as it is in the provided image , like the tags label structure etc. 
    8. Negative_prompts must list what to avoid.
    9. Provide export_notes (dpi/versions) and one-line contrast_guidance if text might overlap textured areas.
    10. All percent values are fractions of canvas height/width as specified.

    SCHEMA for "candidate" (required keys):
    {{
    "id": "string_short_label",
    "style": "editorial_clean|lifestyle|textured|flat_graphic|luxury_minimal",
    "background_prompt": "string (detailed, photo style + props + lighting )",
    "product_Situation_Description" : "string (detailed , How the Product is Placed , wether stand alone at an angle or at a perspective , wether a model is wearing it or using it )"
    "negative_prompts": "string",
    "product_scale": float,                        // 0.2 - 0.9
    "product_position": {{"x_pct": float, "y_pct": float}}, // 0.0 - 1.0
    "product_rotation": float,                     // -45 - 45
    "product_box_hint": {{"x_pct": float, "y_pct": float, "w_pct": float, "h_pct": float}},
    "headline": "string (<=6 words)",
    "headline_position": "top-left|top-right|center-top|center|bottom-left|bottom-right|over_blank_area",
    "headline_font_primary": "string",
    "headline_font_secondary": "string",
    "headline_size_pct": float,                    // fraction of canvas height
    "headline_color": "#hex",
    "subcopy": "string (<=10 words) | empty string",
    "subcopy_size_pct": float,
    "cta_text": "string (<=3 words)",
    "cta_position": "bottom-right|bottom-left|under_headline|inline",
    "cta_font": "string",
    "cta_size_pct": float,
    "cta_button_style": {{"radius_px": int, "padding_pct": float, "bg_color": "#hex", "text_color": "#hex"}},
    "palette": ["#hex1","#hex2","#hex3"],
    "props": ["prop1","prop2"],
    "lighting_directive": "string",
    "texture_or_treatment": "string",
    "contrast_guidance": "string",
    "safe_text_margin_pct": float,
    "export_notes": "string",
    "confidence": float
    }}

    Inputs provided: image (attached or image_url), optional "user_request" (short constraint), optional "audience" and "brand_tone". Use them to bias creative decisions.

    END.
    """
    print("starting the parse_image_parameters node-------")
    
    with open(state['product_image_path'], 'rb') as i:
        input_data = i.read()
    timestamp = int(time.time())
    
    upload_result = cloudinary.uploader.upload(
        input_data,
        public_id=f"qubrid_{timestamp}",
        folder="Adforge",
        resource_type="image"
    )
    print("uploaded the Image-------")
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": The_Prompt},
                    {"type": "input_image", "image_url": upload_result['secure_url']}
                ]
            }]
        )
        
        print("Got the response from the gpt-4.1")
        print(response)
    except Exception as e:
        return {"error": f"Director API call failed: {e}"}

    # --- Extract text from the response object correctly ---
    raw_text = ""
    try:
        # The response structure is: response.output[0].content[0].text
        if hasattr(response, 'output') and response.output:
            first_output = response.output[0]
            if hasattr(first_output, 'content') and first_output.content:
                first_content = first_output.content[0]
                if hasattr(first_content, 'text'):
                    raw_text = first_content.text
        
        # Fallback: try output_text attribute
        if not raw_text and hasattr(response, 'output_text'):
            raw_text = response.output_text
            
        if not raw_text:
            return {"error": f"Could not extract text from response: {response}"}
            
    except Exception as e:
        return {"error": f"Response parsing error: {e}"}

    print(f"Extracted raw_text: {raw_text[:200]}...")

    # --- clean fences and try JSON parse (with small fallback) ---
    clean = raw_text.replace("```json", "").replace("```", "").strip()
    candidate = None
    try:
        data = json.loads(clean)
        candidate = data.get("candidate") if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # fallback: try to find a JSON object via first '{'..'}' slice
        try:
            start = clean.index("{")
            end = clean.rindex("}") + 1
            snippet = clean[start:end]
            data = json.loads(snippet)
            candidate = data.get("candidate") if isinstance(data, dict) else None
        except Exception as e:
            return {"error": f"Director JSON parse failed: {e}. Raw excerpt: {clean[:400]}"}

    if not candidate or not isinstance(candidate, dict):
        return {"error": "Director returned no candidate object or invalid format."}

    # ...existing code...

    required = [
        "style","background_prompt","product_Situation_Description",
        "product_scale","product_position","product_rotation","product_box_hint",
        "headline","headline_position","headline_font_primary","headline_font_secondary","headline_size_pct","headline_color",
        "subcopy","subcopy_size_pct",
        "cta_text","cta_position","cta_font","cta_size_pct","cta_button_style",
        "negative_prompts","palette","props","lighting_directive","texture_or_treatment","contrast_guidance",
        "safe_text_margin_pct","export_notes","confidence"
    ]


    missing = [k for k in required if k not in candidate]
    if missing:
        return {"error": f"Missing keys in candidate: {missing}"}

    # --- coerce & clamp numeric values into safe ranges ---
    # product_scale 0.2-0.9, rotation -45..45, x/y 0..1, sizes as earlier
    product_scale = clamp(candidate.get("product_scale", 0.6), 0.2, 0.9)
    product_rotation = clamp(candidate.get("product_rotation", 0.0), -45.0, 45.0)

    pos = candidate.get("product_position", {})
    if not isinstance(pos, dict):
        pos = {"x_pct": 0.5, "y_pct": 0.5}
    x_pct = clamp(pos.get("x_pct", 0.5), 0.0, 1.0)
    y_pct = clamp(pos.get("y_pct", 0.5), 0.0, 1.0)

    headline_size_pct = clamp(candidate.get("headline_size_pct", 0.06), 0.02, 0.15)
    subcopy_size_pct = clamp(candidate.get("subcopy_size_pct", 0.03), 0.012, 0.06) if candidate.get("subcopy") else None
    cta_size_pct = clamp(candidate.get("cta_size_pct", 0.03), 0.018, 0.08)
    safe_text_margin_pct = clamp(candidate.get("safe_text_margin_pct", 0.03), 0.0, 0.2)
    confidence = clamp(candidate.get("confidence", 0.9), 0.0, 1.0)

    # --- final return mapping (types normalized) ---
    try:
        return {
            "style": candidate.get("style"),
            "background_prompt": candidate["background_prompt"],
            "product_Situation_Description": candidate["product_Situation_Description"],
            "negative_prompts": candidate.get("negative_prompts"),

            "headline": candidate["headline"],
            "headline_position": candidate["headline_position"],
            "headline_font_primary": candidate["headline_font_primary"],
            "headline_font_secondary": candidate["headline_font_secondary"],
            "headline_size_pct": headline_size_pct,
            "headline_color": candidate["headline_color"],

            "subcopy": candidate.get("subcopy", ""),
            "subcopy_size_pct": subcopy_size_pct,

            "cta_text": candidate["cta_text"],
            "cta_position": candidate["cta_position"],
            "cta_font": candidate["cta_font"],
            "cta_size_pct": cta_size_pct,
            "cta_button_style": candidate["cta_button_style"],

            "palette": candidate["palette"],
            "props": candidate.get("props", []),
            "lighting_directive": candidate.get("lighting_directive"),
            "texture_or_treatment": candidate.get("texture_or_treatment"),
            "contrast_guidance": candidate.get("contrast_guidance"),

            "product_scale": product_scale,
            "product_position": {"x_pct": x_pct, "y_pct": y_pct},
            "product_rotation": product_rotation,
            "product_box_hint": candidate.get("product_box_hint"),

            "safe_text_margin_pct": safe_text_margin_pct,
            "export_notes": candidate.get("export_notes"),

            "confidence": confidence,

            "retry_count": 0,
            "quality_score": 0,
            "feedback_logs": [],
            "best_score": -1,
            "best_image_url": None,
            "error": None
        }

    except Exception as e:
        return {"error": f"Mapping failure: {e}"}


            
def parsing_image(state: AdState):
    state.setdefault("image_url_for_AD", None)
    state.setdefault("best_score", -1)
    state.setdefault("best_image_url", None)
    state.setdefault('retry_count', 0)
    state.setdefault('feedback_logs', [])
    target_w = state.get('target_width', 1024)
    target_h = state.get('target_height', 1024)

    print(f"🛠️ Processing: {state['product_image_path']} to {target_w}x{target_h}...")
    try:
        with open(state['product_image_path'], 'rb') as i:
            input_data = i.read()

        product_rotation = float(state.get("product_rotation", 0))
        #product_rotation = 0
        #product_position = {"x_pct": 0.5, "y_pct": 0.5}
        product_position = state.get(
            "product_position",
            {"x_pct": 0.5, "y_pct": 0.5} 
        )
        product_scale = float(state.get("product_scale", 0.6))
        #product_scale = 1.0
        output_data = remove(input_data)
        product_img = Image.open(io.BytesIO(output_data)).convert("RGBA")

        # Canvas
        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))

        
        scale_factor = (max(target_w, target_h) * product_scale) / min(product_img.size)
        new_size = (
            int(product_img.width * scale_factor),
            int(product_img.height * scale_factor)
        )
        product_resized = product_img.resize(new_size, Image.Resampling.LANCZOS)

        # 🔄 Rotate product
        if product_rotation != 0:
            product_resized = product_resized.rotate(
                product_rotation,
                resample=Image.BICUBIC,
                expand=True
            )

        
        rot_w, rot_h = product_resized.size

        offset_x = int((target_w - rot_w) * product_position["x_pct"])
        offset_y = int((target_h - rot_h) * product_position["y_pct"])

        canvas.paste(product_resized, (offset_x, offset_y), product_resized)

        # Save Source
        source_filename = "/tmp/temp_source.png"
        canvas.save(source_filename, format="PNG")
                
        
        mask = Image.new("L", (target_w, target_h), 255)
        canvas_alpha = canvas.split()[-1]
        product_area_mask = ImageOps.invert(canvas_alpha)
        mask.paste(product_area_mask, (0, 0), canvas_alpha)
        
        mask_filename = "/tmp/temp_mask.png"
        mask.save(mask_filename, format="PNG")
        
        return {
            "source_filename": source_filename,
            "mask_filename": mask_filename
        }
    except Exception as e:
        return {"error": f"Parsing Error: {str(e)}"}

def generate_background_image(state: AdState):
    if state.get("error"): return {"background_image_path": None}
    
    prompt_text = f"""{state['background_prompt']} and {state['product_Situation_Description']} and 
    the style is :{state['style']} 
    and product_scale : {state["product_scale"]} (MAKE SURE TO DO) where scale is 0-1 | product_rotation : {state["product_rotation"]} (MAKE SURE  TO ROTATE IF THERE IS ANY VALUE HERE) |
    product_position : {state["product_position"]} | product_box_hint : {state["product_box_hint"]}
    safe_text_margin_pct : {state["safe_text_margin_pct"]}
    props : {state["props"]} | lighting_directive : {state["lighting_directive"]}
    texture_or_treatment : {state["texture_or_treatment"]} | contrast_guidance : {state["contrast_guidance"]}
    you have to make the image look like all the structure according to the 
    given instruction and MAKE SURE , DON'T CHANGE THE PRODUCT BRAND AND LABEL AND ILLUSTRATIONS ON IT"""
    print(f"🎨 Generating Background: {prompt_text}...")
    
    try:
        
            response = openai_client.images.edit(
                model="gpt-image-1.5",
                image=open(state['source_filename'], "rb"),
                prompt=prompt_text,
                n=1,
                size=f"{int(state['target_width'])}x{int(state['target_height'])}" 
            )
            base64_string = response.data[0].b64_json
            b64 = response.data[0].b64_json
            if not b64:
                    return {"error": "No image returned from OpenAI"}
            if base64_string:      
                    timestamp = int(time.time())
                    upload_result = cloudinary.uploader.upload(
                        f"data:image/png;base64,{base64_string}",
                        public_id=f"qubrid_{timestamp}",
                        folder="Adforge"
                    )
                    
                    return {
                        "image_url_for_background": upload_result['secure_url'],
                        "cloudinary_id": upload_result['public_id']
                    }
            else:
                    return {"error": f"API Error: {response}"}
    except Exception as e:
            return {"error": str(e)}
        
def generate_Ad_image(state: AdState):
    print(" 😶‍🌫️ Here we come into the Ad image node-----")
    if state.get("error"): return {"final_image_path": None}
    if not state.get('image_url_for_background'): return {"error": "Missing background"}  
    
    instructions = f"""add the headline : {state["headline"]} | headline_position : {state["headline_position"]}
    headline_font_primary : {state["headline_font_primary"]} | headline_font_secondary : {state["headline_font_secondary"]}
    headline_size_pct : {state["headline_size_pct"]} | headline_color : {state["headline_color"]}
    subcopy : {state["subcopy"]} | subcopy_size_pct : {state["subcopy_size_pct"]} and the cta_text : {state["cta_text"]} | cta_position : {state["cta_position"]}
    cta_font : {state["cta_font"]} | cta_size_pct : {state["cta_size_pct"]}
    cta_button_style : {state["cta_button_style"]} , palette : {state["palette"]} | and product_scale : {state["product_scale"]} | product_rotation : {state["product_rotation"]}
    product_position : {state["product_position"]} | product_box_hint : {state["product_box_hint"]}
    safe_text_margin_pct : {state["safe_text_margin_pct"]}
    props : {state["props"]} | lighting_directive : {state["lighting_directive"]}
    texture_or_treatment : {state["texture_or_treatment"]} | contrast_guidance : {state["contrast_guidance"]} """
       
    current_retry = state.get("retry_count", 0)
    feedback_history = state.get("feedback_logs", [])
    print (f"The ADD Generation Text {instructions}")

    if current_retry > 0 and feedback_history:
        last_error = feedback_history[-1]
        print(f"⚠️ RETRY MODE DETECTED ({current_retry}/2). Fixing: {last_error[:50]}...")
             
        instructions = f" CRITICAL INSTRUCTION: '{last_error}'. YOU MUST FIX THIS SPECIFIC ERROR in this version."

        print(f"🎨 Generating Ad (Attempt {current_retry + 1}): {instructions[:50]}...")
        
        try:
            download_response = requests.get(state['image_url_for_AD'])
            if download_response.status_code != 200: return {"error": "Failed to download bg"}
            
            bg_file = BytesIO(download_response.content)
            bg_file.name = "background.png"  # give it a filename so the SDK recognizes it

            response = openai_client.images.edit(
                model="gpt-image-1.5",
                prompt=instructions,
                image = bg_file,
                n=1,
                size=f"{int(state['target_width'])}x{int(state['target_height'])}" 
            )
            base64_string = response.data[0].b64_json
            if base64_string:      
                    timestamp = int(time.time())
                    upload_result = cloudinary.uploader.upload(
                        f"data:image/png;base64,{base64_string}",
                        public_id=f"qubrid_{timestamp}",
                        folder="Adforge"
                    )
                    
                    return {
                        "image_url_for_AD": upload_result['secure_url'],
                        "cloudinary_id": upload_result['public_id']
                    }
            else:
                    return {"error": f"API Error: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    else:
        
        try:
            download_response = requests.get(state['image_url_for_background'])
            if download_response.status_code != 200: return {"error": "Failed to download bg"}
            
            bg_file = BytesIO(download_response.content)
            bg_file.name = "background1.png" 

            response = openai_client.images.edit(
                model="gpt-image-1.5",
                prompt=instructions,
                image = bg_file,
                n=1,
                size=f"{int(state['target_width'])}x{int(state['target_height'])}"
            )
            base64_string = response.data[0].b64_json
            if base64_string:      
                    timestamp = int(time.time())
                    upload_result = cloudinary.uploader.upload(
                        f"data:image/png;base64,{base64_string}",
                        public_id=f"qubrid_{timestamp}",
                        folder="Adforge"
                    )
                    
                    return {
                        "image_url_for_AD": upload_result['secure_url'],
                        "cloudinary_id": upload_result['public_id']
                    }
            else:
                    return {"error": f"API Error: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
        
def VisionModel(state: AdState):
    if not state.get("image_url_for_AD"):
        return {"error": "Missing ad image URL for review"}
    Vision_Prompt = f"""
    Review this image against:headline : {state["headline"]} | headline_position : {state["headline_position"]}
headline_font_primary : {state["headline_font_primary"]} | headline_font_secondary : {state["headline_font_secondary"]}
headline_size_pct : {state["headline_size_pct"]} | headline_color : {state["headline_color"]}
subcopy : {state["subcopy"]} | subcopy_size_pct : {state["subcopy_size_pct"]}

cta_text : {state["cta_text"]} | cta_position : {state["cta_position"]}
cta_font : {state["cta_font"]} | cta_size_pct : {state["cta_size_pct"]}
cta_button_style : {state["cta_button_style"]}

style : {state["style"]}
background_prompt : {state["background_prompt"]} | negative_prompts : {state["negative_prompts"]}

palette : {state["palette"]} | 
product_scale : {state["product_scale"]} | product_rotation : {state["product_rotation"]}
product_position : {state["product_position"]} | product_box_hint : {state["product_box_hint"]}
safe_text_margin_pct : {state["safe_text_margin_pct"]}

props : {state["props"]} | lighting_directive : {state["lighting_directive"]}
texture_or_treatment : {state["texture_or_treatment"]} | contrast_guidance : {state["contrast_guidance"]} .
    Output strictly JSON:
    "ImageRating": Rate 1-10 (Be harsh. If any text is misspelled or jumbled, score must be < 10).
    "Potential_errors": Instruct exactly to the point (not any if and buts or anything rubbish , you have to tell the instruction like a human would do, cause your this given instruction would go to the image editing model for the fixes) what is wrong for example , There could be spelling mistake like shop is spelled like shoop , or maybe any of the text could be missing those I have given you in the prompt as well, or maybe the words migh not be clear.
    """

    print(f"👁️ Reviewing Image...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(ImageReview)
   
    try:
        # --- 4. Invoke the Chain ---
        # LangChain handles the image input via HumanMessage content list
        message = HumanMessage(
            content=[
                {"type": "text", "text": Vision_Prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": state['image_url_for_AD']},
                },
            ]
        )

        # The result is directly an instance of the Pydantic class 'ImageReview'
        result: ImageReview = structured_llm.invoke([message])

        # --- 5. Process Result ---
        new_rating = result.ImageRating
        new_feedback = result.Potential_errors
        print(f"The Potential Erros in the {state['retry_count']} Attempt {new_feedback}")
        
        print(f"✅ Vision Score: {new_rating}/10 | Feedback: {new_feedback}")

        # Update State logic
        current_logs = state.get("feedback_logs", [])
        if new_feedback:
            current_logs.extend(new_feedback)

        best_score = state.get("best_score", -1)
        best_image_url = state.get("best_image_url", "")
        
        if new_rating > best_score:
            best_score = new_rating
            best_image_url = state["image_url_for_AD"]

        return {
            "quality_score": new_rating,
            "feedback_logs": current_logs,
            "retry_count": state.get("retry_count", 0) + 1,
            "best_score": best_score,
            "best_image_url": best_image_url,
        }

    except Exception as e:
        print(f"❌ Error in VisionModel: {e}")
        return {"error": str(e)}



def check_quality(state: AdState):
    score = state.get("quality_score", 0)
    retries = state.get("retry_count", 0)
    MAX_RETRIES = 2

    print(f"🔄 Check: Score {score}/10 | Attempts: {retries}/{MAX_RETRIES}")
    
    # --- FIX IS HERE: Use .get() to avoid KeyError ---
    current_ad_url = state.get('image_url_for_AD', "No URL generated yet")
    print(f"Ad Link of the score {score} is : {current_ad_url}")
    
    # Condition: If score is perfect OR we ran out of retries -> END
    if score >= 10 or retries >= MAX_RETRIES:
        if retries >= MAX_RETRIES:
            print("🛑 Max retries reached. Stopping.")
        else:
            print("🎉 Perfect Score! Stopping.")
        return END
    
    else:
        print("🔙 Quality low. Sending back to Generator...")
        return "Text_For_AD"

# --- Our Graph Setup ---
graph = StateGraph(AdState)

graph.add_node("Generating_Node", parsing_image_parameters)
graph.add_node("parsing", parsing_image)
graph.add_node("Generator", generate_background_image)
graph.add_node("Text_For_AD", generate_Ad_image)
graph.add_node("VisionNode", VisionModel)
graph.add_edge(START, "Generating_Node")
graph.add_edge("Generating_Node", "parsing")
graph.add_edge("parsing", "Generator")
graph.add_edge("Generator", "Text_For_AD")
graph.add_edge("Text_For_AD", "VisionNode")

graph.add_conditional_edges(
    "VisionNode",       # Start from Vision Node
    check_quality,      # Run this function to decide where to go
    {                   # Map the function return values to Nodes
        END: END,
        "Text_For_AD": "Text_For_AD"
    }
)

app = graph.compile()