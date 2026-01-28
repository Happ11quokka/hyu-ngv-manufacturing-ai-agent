"""
dAIso - Semiconductor Defect Detection Demo Interface
Agent V10: YOLO + Two-Stage GPT-4o Pipeline
"""

import gradio as gr
import json
import os
from pathlib import Path
import random

# Load dummy data
DATA_PATH = Path(__file__).parent / "data" / "dummy_results.json"

def load_data():
    """Load the dummy results data."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

DATA = load_data()

# ============================================================
# 1. Dashboard / Statistics Tab
# ============================================================

def get_dashboard_stats():
    """Generate dashboard statistics HTML."""
    summary = DATA["summary"]
    defects = DATA["defect_statistics"]

    stats_html = f"""
    <div style="padding: 20px;">
        <h2 style="color: #2563eb; margin-bottom: 20px;">Agent V10 Analysis Summary</h2>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); padding: 20px; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 36px; font-weight: bold;">{summary['total_analyzed']}</div>
                <div style="opacity: 0.9;">Total Analyzed</div>
            </div>
            <div style="background: linear-gradient(135deg, #22c55e, #16a34a); padding: 20px; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 36px; font-weight: bold;">{summary['normal_count']}</div>
                <div style="opacity: 0.9;">Normal</div>
            </div>
            <div style="background: linear-gradient(135deg, #ef4444, #dc2626); padding: 20px; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 36px; font-weight: bold;">{summary['abnormal_count']}</div>
                <div style="opacity: 0.9;">Abnormal</div>
            </div>
            <div style="background: linear-gradient(135deg, #8b5cf6, #7c3aed); padding: 20px; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 36px; font-weight: bold;">{summary['accuracy']*100:.0f}%</div>
                <div style="opacity: 0.9;">Accuracy</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                <div style="color: #64748b; font-size: 14px;">Average Confidence</div>
                <div style="font-size: 28px; font-weight: bold; color: #1e293b;">{summary['avg_confidence']*100:.1f}%</div>
            </div>
            <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                <div style="color: #64748b; font-size: 14px;">Avg Processing Time</div>
                <div style="font-size: 28px; font-weight: bold; color: #1e293b;">{summary['avg_processing_time_ms']}ms</div>
            </div>
            <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                <div style="color: #64748b; font-size: 14px;">Rechecks Triggered</div>
                <div style="font-size: 28px; font-weight: bold; color: #1e293b;">{summary['recheck_triggered']}</div>
            </div>
        </div>

        <h3 style="color: #1e293b; margin-bottom: 15px;">Detected Defect Types</h3>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
    """

    defect_colors = {
        "bent_lead": "#f59e0b",
        "crossed_leads": "#ef4444",
        "missed_hole": "#dc2626",
        "body_tilt": "#8b5cf6",
        "surface_contamination": "#6366f1",
        "lead_deformation": "#ec4899",
        "floating_lead": "#f43f5e"
    }

    for defect, count in defects.items():
        color = defect_colors.get(defect, "#64748b")
        display_name = defect.replace("_", " ").title()
        stats_html += f"""
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid {color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 24px; font-weight: bold; color: {color};">{count}</div>
                <div style="color: #64748b; font-size: 13px;">{display_name}</div>
            </div>
        """

    stats_html += """
        </div>
    </div>
    """

    return stats_html


def get_pie_chart_data():
    """Get data for the pie chart visualization."""
    summary = DATA["summary"]
    return {
        "Normal": summary["normal_count"],
        "Abnormal": summary["abnormal_count"]
    }


def get_defect_bar_data():
    """Get data for defect type bar chart."""
    return DATA["defect_statistics"]


# ============================================================
# 2. Individual Results Tab
# ============================================================

def get_image_list():
    """Get list of all image IDs."""
    return [r["id"] for r in DATA["results"]]


def get_result_detail(image_id):
    """Get detailed result for a specific image."""
    for result in DATA["results"]:
        if result["id"] == image_id:
            return result
    return None


def format_result_detail(image_id):
    """Format detailed result as HTML."""
    result = get_result_detail(image_id)
    if not result:
        return "Image not found", None

    # Get local image path
    image_path = Path(__file__).parent / "data" / result["image_path"]

    # Status badge
    status_color = "#22c55e" if result["label"] == 0 else "#ef4444"
    status_text = result["label_text"]

    # YOLO detection info
    yolo = result["yolo_detection"]

    # Stage 1 observation
    obs = result["stage1_observation"]
    body = obs["body"]
    leads = obs["leads"]

    # Stage 2 decision
    decision = result["stage2_decision"]

    html = f"""
    <div style="padding: 20px; background: #f8fafc; border-radius: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1e293b;">{result['id']}</h2>
            <div style="display: flex; gap: 10px; align-items: center;">
                <span style="background: {status_color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold;">
                    {status_text}
                </span>
                <span style="background: #3b82f6; color: white; padding: 8px 16px; border-radius: 20px;">
                    Confidence: {result['confidence']*100:.1f}%
                </span>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <!-- Left Column: YOLO Detection -->
            <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3 style="color: #2563eb; margin-top: 0;">Stage 0: YOLO Detection</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px 0; color: #64748b;">Holes Detected</td>
                        <td style="padding: 8px 0; font-weight: bold;">{yolo['holes']['count']} (conf: {yolo['holes']['confidence_avg']*100:.0f}%)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px 0; color: #64748b;">Leads Detected</td>
                        <td style="padding: 8px 0; font-weight: bold;">{yolo['leads']['count']} (conf: {yolo['leads']['confidence_avg']*100:.0f}%)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #64748b;">Body Detected</td>
                        <td style="padding: 8px 0; font-weight: bold;">{yolo['body']['count']} (conf: {yolo['body']['confidence']*100:.0f}%)</td>
                    </tr>
                </table>
            </div>

            <!-- Right Column: Processing Info -->
            <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3 style="color: #2563eb; margin-top: 0;">Processing Info</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px 0; color: #64748b;">Processing Time</td>
                        <td style="padding: 8px 0; font-weight: bold;">{result['processing_time_ms']}ms</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px 0; color: #64748b;">Recheck Performed</td>
                        <td style="padding: 8px 0; font-weight: bold;">{'Yes' if result['recheck_performed'] else 'No'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #64748b;">Recheck Type</td>
                        <td style="padding: 8px 0; font-weight: bold;">{result.get('recheck_type', 'N/A')}</td>
                    </tr>
                </table>
            </div>
        </div>

        <!-- Stage 1: Visual Observation -->
        <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 20px;">
            <h3 style="color: #16a34a; margin-top: 0;">Stage 1: GPT-4o Visual Observation</h3>

            <div style="margin-bottom: 15px;">
                <h4 style="color: #1e293b; margin-bottom: 10px;">Body Analysis</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                    <div style="background: #f1f5f9; padding: 10px; border-radius: 6px; text-align: center;">
                        <div style="color: #64748b; font-size: 12px;">Tilt</div>
                        <div style="font-weight: bold;">{body['tilt']}</div>
                    </div>
                    <div style="background: #f1f5f9; padding: 10px; border-radius: 6px; text-align: center;">
                        <div style="color: #64748b; font-size: 12px;">Rotation</div>
                        <div style="font-weight: bold;">{body['rotation']}°</div>
                    </div>
                    <div style="background: #f1f5f9; padding: 10px; border-radius: 6px; text-align: center;">
                        <div style="color: #64748b; font-size: 12px;">Center Offset</div>
                        <div style="font-weight: bold;">{body['center_offset']}</div>
                    </div>
                    <div style="background: #f1f5f9; padding: 10px; border-radius: 6px; text-align: center;">
                        <div style="color: #64748b; font-size: 12px;">Surface Marks</div>
                        <div style="font-weight: bold;">{'Yes' if body['surface_marks'] else 'No'}</div>
                    </div>
                </div>
            </div>

            <div style="margin-bottom: 15px;">
                <h4 style="color: #1e293b; margin-bottom: 10px;">Lead Analysis</h4>
                <table style="width: 100%; border-collapse: collapse; background: #f8fafc; border-radius: 8px; overflow: hidden;">
                    <thead>
                        <tr style="background: #e2e8f0;">
                            <th style="padding: 12px; text-align: left;">Lead</th>
                            <th style="padding: 12px; text-align: left;">Visibility</th>
                            <th style="padding: 12px; text-align: left;">Shape</th>
                            <th style="padding: 12px; text-align: left;">Position</th>
                            <th style="padding: 12px; text-align: left;">Contact</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for lead in leads:
        contact_color = "#22c55e" if lead["contact"] else "#ef4444"
        shape_color = "#22c55e" if lead["shape"] == "straight" else "#f59e0b"
        html += f"""
                        <tr style="border-bottom: 1px solid #e2e8f0;">
                            <td style="padding: 12px; font-weight: bold;">{lead['id'].title()}</td>
                            <td style="padding: 12px;">{lead['visibility']}</td>
                            <td style="padding: 12px;"><span style="color: {shape_color}; font-weight: bold;">{lead['shape']}</span></td>
                            <td style="padding: 12px;">{lead['end_position']}</td>
                            <td style="padding: 12px;"><span style="color: {contact_color}; font-weight: bold;">{'Yes' if lead['contact'] else 'No'}</span></td>
                        </tr>
        """

    arrangement_color = "#22c55e" if obs["lead_arrangement"] == "parallel" else "#f59e0b"
    html += f"""
                    </tbody>
                </table>
                <div style="margin-top: 10px; display: flex; gap: 20px;">
                    <div>Lead Arrangement: <span style="color: {arrangement_color}; font-weight: bold;">{obs['lead_arrangement']}</span></div>
                    <div>All Leads Reach Holes: <span style="color: {'#22c55e' if obs['all_leads_reach_holes'] else '#ef4444'}; font-weight: bold;">{'Yes' if obs['all_leads_reach_holes'] else 'No'}</span></div>
                </div>
            </div>
        </div>

        <!-- Stage 2: Decision -->
        <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 20px;">
            <h3 style="color: #dc2626; margin-top: 0;">Stage 2: GPT-4o Decision</h3>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 15px;">
                <div style="background: #fef2f2; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #dc2626;">{decision['defect_score']}</div>
                    <div style="color: #991b1b; font-size: 12px;">Defect Score</div>
                </div>
                <div style="background: #fef2f2; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #dc2626;">{len(decision['critical_defects'])}</div>
                    <div style="color: #991b1b; font-size: 12px;">Critical</div>
                </div>
                <div style="background: #fff7ed; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #ea580c;">{len(decision['high_defects'])}</div>
                    <div style="color: #9a3412; font-size: 12px;">High</div>
                </div>
                <div style="background: #fefce8; padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 28px; font-weight: bold; color: #ca8a04;">{len(decision['medium_defects'])}</div>
                    <div style="color: #854d0e; font-size: 12px;">Medium</div>
                </div>
            </div>
    """

    if decision['critical_defects']:
        html += f"""
            <div style="background: #fef2f2; padding: 10px 15px; border-radius: 6px; margin-bottom: 10px;">
                <strong style="color: #dc2626;">Critical Defects:</strong> {', '.join(decision['critical_defects'])}
            </div>
        """
    if decision['high_defects']:
        html += f"""
            <div style="background: #fff7ed; padding: 10px 15px; border-radius: 6px; margin-bottom: 10px;">
                <strong style="color: #ea580c;">High Defects:</strong> {', '.join(decision['high_defects'])}
            </div>
        """
    if decision['medium_defects']:
        html += f"""
            <div style="background: #fefce8; padding: 10px 15px; border-radius: 6px; margin-bottom: 10px;">
                <strong style="color: #ca8a04;">Medium Defects:</strong> {', '.join(decision['medium_defects'])}
            </div>
        """

    html += """
        </div>

        <!-- Reasons -->
        <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 20px;">
            <h3 style="color: #1e293b; margin-top: 0;">Analysis Reasons</h3>
            <ul style="margin: 0; padding-left: 20px;">
    """

    for reason in result["reasons"]:
        html += f"<li style='margin-bottom: 8px; color: #475569;'>{reason}</li>"

    html += """
            </ul>
        </div>
    </div>
    """

    return html, str(image_path)


# ============================================================
# 3. Visualization Tab
# ============================================================

def create_classification_chart():
    """Create classification distribution chart data."""
    import matplotlib.pyplot as plt
    import io
    import base64

    summary = DATA["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    labels = ['Normal', 'Abnormal']
    sizes = [summary['normal_count'], summary['abnormal_count']]
    colors = ['#22c55e', '#ef4444']
    explode = (0.05, 0.05)

    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
    axes[0].set_title('Classification Distribution', fontsize=14, fontweight='bold')

    # Bar chart for defects
    defects = DATA["defect_statistics"]
    defect_names = [name.replace('_', '\n') for name in defects.keys()]
    defect_counts = list(defects.values())
    colors_bar = ['#f59e0b', '#ef4444', '#dc2626', '#8b5cf6', '#6366f1', '#ec4899', '#f43f5e']

    bars = axes[1].bar(defect_names, defect_counts, color=colors_bar[:len(defect_names)])
    axes[1].set_title('Defect Types Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')

    for bar, count in zip(bars, defect_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


def create_confidence_chart():
    """Create confidence distribution chart."""
    import matplotlib.pyplot as plt
    import io

    results = DATA["results"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confidence distribution
    confidences = [r["confidence"] for r in results]
    normal_conf = [r["confidence"] for r in results if r["label"] == 0]
    abnormal_conf = [r["confidence"] for r in results if r["label"] == 1]

    axes[0].hist([normal_conf, abnormal_conf], bins=10, label=['Normal', 'Abnormal'],
                 color=['#22c55e', '#ef4444'], alpha=0.7, edgecolor='white')
    axes[0].set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Processing time distribution
    proc_times = [r["processing_time_ms"] for r in results]
    recheck_yes = [r["processing_time_ms"] for r in results if r["recheck_performed"]]
    recheck_no = [r["processing_time_ms"] for r in results if not r["recheck_performed"]]

    axes[1].boxplot([recheck_no, recheck_yes], labels=['No Recheck', 'Recheck'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#3b82f6', alpha=0.7),
                    medianprops=dict(color='#1e293b', linewidth=2))
    axes[1].set_title('Processing Time by Recheck Status', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Processing Time (ms)')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


def create_yolo_performance_chart():
    """Create YOLO detection performance chart."""
    import matplotlib.pyplot as plt
    import io

    results = DATA["results"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # YOLO detection confidence
    hole_conf = [r["yolo_detection"]["holes"]["confidence_avg"] for r in results]
    lead_conf = [r["yolo_detection"]["leads"]["confidence_avg"] for r in results]
    body_conf = [r["yolo_detection"]["body"]["confidence"] for r in results]

    x = range(len(results))

    axes[0].plot(x, hole_conf, 'o-', label='Holes', color='#3b82f6', markersize=6)
    axes[0].plot(x, lead_conf, 's-', label='Leads', color='#22c55e', markersize=6)
    axes[0].plot(x, body_conf, '^-', label='Body', color='#f59e0b', markersize=6)
    axes[0].set_title('YOLO Detection Confidence per Image', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Image Index')
    axes[0].set_ylabel('Confidence')
    axes[0].legend()
    axes[0].set_ylim(0.7, 1.0)
    axes[0].grid(True, alpha=0.3)

    # Hole count distribution
    hole_counts = [r["yolo_detection"]["holes"]["count"] for r in results]
    axes[1].hist(hole_counts, bins=range(15, 22), color='#3b82f6', alpha=0.7, edgecolor='white')
    axes[1].set_title('Detected Holes Count Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Holes Detected')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


# ============================================================
# 4. Chatbot Tab
# ============================================================

def get_chatbot_response(message, history):
    """Generate response for the chatbot based on data analysis."""
    message_lower = message.lower()

    # Analyze the question and provide relevant response
    if any(word in message_lower for word in ["전체", "총", "summary", "요약", "개요"]):
        summary = DATA["summary"]
        response = f"""**Analysis Summary:**
- Total Images Analyzed: {summary['total_analyzed']}
- Normal: {summary['normal_count']} ({summary['normal_count']/summary['total_analyzed']*100:.1f}%)
- Abnormal: {summary['abnormal_count']} ({summary['abnormal_count']/summary['total_analyzed']*100:.1f}%)
- Average Confidence: {summary['avg_confidence']*100:.1f}%
- Average Processing Time: {summary['avg_processing_time_ms']}ms
- Rechecks Triggered: {summary['recheck_triggered']}"""

    elif any(word in message_lower for word in ["결함", "defect", "불량", "이상"]):
        defects = DATA["defect_statistics"]
        response = "**Detected Defect Types:**\n"
        for defect, count in sorted(defects.items(), key=lambda x: -x[1]):
            display_name = defect.replace("_", " ").title()
            response += f"- {display_name}: {count} cases\n"

        # Find most common defect
        most_common = max(defects.items(), key=lambda x: x[1])
        response += f"\n**Most Common Defect:** {most_common[0].replace('_', ' ').title()} ({most_common[1]} cases)"

    elif any(word in message_lower for word in ["정상", "normal", "양품"]):
        normal_results = [r for r in DATA["results"] if r["label"] == 0]
        response = f"**Normal Samples: {len(normal_results)} images**\n\n"
        response += "| ID | Confidence | Processing Time |\n|---|---|---|\n"
        for r in normal_results[:10]:
            response += f"| {r['id']} | {r['confidence']*100:.1f}% | {r['processing_time_ms']}ms |\n"
        if len(normal_results) > 10:
            response += f"\n... and {len(normal_results) - 10} more"

    elif any(word in message_lower for word in ["비정상", "abnormal", "불량품"]):
        abnormal_results = [r for r in DATA["results"] if r["label"] == 1]
        response = f"**Abnormal Samples: {len(abnormal_results)} images**\n\n"
        for r in abnormal_results:
            reasons = ", ".join(r["reasons"][:2])
            response += f"- **{r['id']}** (Confidence: {r['confidence']*100:.1f}%): {reasons}\n"

    elif "dev_" in message_lower or any(f"dev_{i:03d}" in message_lower for i in range(20)):
        # Find specific image ID
        import re
        match = re.search(r'dev_\d{3}', message_lower)
        if match:
            image_id = match.group().upper()
            result = get_result_detail(image_id)
            if result:
                response = f"""**{result['id']} Analysis Result:**
- **Classification:** {result['label_text']}
- **Confidence:** {result['confidence']*100:.1f}%
- **Processing Time:** {result['processing_time_ms']}ms
- **Recheck:** {'Yes (' + result.get('recheck_type', 'N/A') + ')' if result['recheck_performed'] else 'No'}

**YOLO Detection:**
- Holes: {result['yolo_detection']['holes']['count']} (conf: {result['yolo_detection']['holes']['confidence_avg']*100:.0f}%)
- Leads: {result['yolo_detection']['leads']['count']} (conf: {result['yolo_detection']['leads']['confidence_avg']*100:.0f}%)

**Reasons:**
"""
                for reason in result["reasons"]:
                    response += f"- {reason}\n"
            else:
                response = f"Image {image_id} not found in the analysis results."
        else:
            response = "Please provide a valid image ID (e.g., DEV_001, DEV_015)."

    elif any(word in message_lower for word in ["confidence", "신뢰도", "확신"]):
        results = DATA["results"]
        high_conf = [r for r in results if r["confidence"] >= 0.9]
        low_conf = [r for r in results if r["confidence"] < 0.85]

        response = f"""**Confidence Analysis:**
- High Confidence (≥90%): {len(high_conf)} images
- Low Confidence (<85%): {len(low_conf)} images
- Average: {DATA['summary']['avg_confidence']*100:.1f}%

**Low Confidence Samples:**
"""
        for r in low_conf:
            response += f"- {r['id']}: {r['confidence']*100:.1f}% ({r['label_text']})\n"

    elif any(word in message_lower for word in ["recheck", "재검사", "검증"]):
        recheck_results = [r for r in DATA["results"] if r["recheck_performed"]]
        response = f"""**Recheck Analysis:**
- Total Rechecks: {len(recheck_results)}
- Recheck Types Used:

"""
        recheck_types = {}
        for r in recheck_results:
            rtype = r.get("recheck_type", "unknown")
            recheck_types[rtype] = recheck_types.get(rtype, 0) + 1

        for rtype, count in recheck_types.items():
            response += f"- {rtype}: {count} times\n"

        response += "\n**Images with Recheck:**\n"
        for r in recheck_results:
            response += f"- {r['id']}: {r.get('recheck_type', 'N/A')} → {r['label_text']}\n"

    elif any(word in message_lower for word in ["yolo", "detection", "탐지"]):
        results = DATA["results"]
        avg_holes = sum(r["yolo_detection"]["holes"]["count"] for r in results) / len(results)
        avg_hole_conf = sum(r["yolo_detection"]["holes"]["confidence_avg"] for r in results) / len(results)
        avg_lead_conf = sum(r["yolo_detection"]["leads"]["confidence_avg"] for r in results) / len(results)
        avg_body_conf = sum(r["yolo_detection"]["body"]["confidence"] for r in results) / len(results)

        response = f"""**YOLO Detection Statistics:**

| Component | Avg Detection | Avg Confidence |
|---|---|---|
| Holes | {avg_holes:.1f} | {avg_hole_conf*100:.1f}% |
| Leads | 3.0 | {avg_lead_conf*100:.1f}% |
| Body | 1.0 | {avg_body_conf*100:.1f}% |

YOLO Stage provides initial component localization before GPT-4o visual analysis."""

    elif any(word in message_lower for word in ["pipeline", "파이프라인", "process", "과정", "단계"]):
        response = """**Agent V10 Pipeline:**

**Stage 0: YOLO Detection**
- Detects holes, leads, and body using Roboflow API
- Provides bounding boxes and confidence scores
- Performs lead-hole alignment analysis

**Stage 1: GPT-4o Visual Observation**
- Analyzes image with YOLO hints as context
- Evaluates body tilt, rotation, surface marks
- Checks lead shape, position, and contact status
- Determines lead arrangement pattern

**Stage 2: GPT-4o Decision Making**
- Converts observations to defect score
- Applies critical/high/medium rules
- Classification: score ≥ 3 = abnormal

**Conditional Recheck:**
- Triggered when confidence < 85% or suspicious findings
- Types: leads_focus, patch_recheck, body_alignment, dual_model
- Final decision via weighted voting"""

    elif any(word in message_lower for word in ["help", "도움", "사용법", "?"]):
        response = """**Available Commands:**

**General Queries:**
- "전체 요약" / "summary" - Overall analysis summary
- "결함 종류" / "defects" - Defect type statistics
- "정상 샘플" / "normal samples" - List of normal samples
- "비정상 샘플" / "abnormal samples" - List of abnormal samples

**Specific Analysis:**
- "DEV_001 결과" - Specific image analysis result
- "confidence 분석" - Confidence score analysis
- "recheck 현황" - Recheck statistics
- "YOLO 성능" - YOLO detection performance

**System Info:**
- "pipeline 설명" - V10 pipeline explanation
- "help" - This help message"""

    else:
        # Default response
        response = """I can help you with the following:

1. **Summary**: "전체 요약", "summary"
2. **Defects**: "결함 종류", "defect types"
3. **Normal samples**: "정상 샘플 목록"
4. **Abnormal samples**: "비정상 샘플 분석"
5. **Specific image**: "DEV_001 결과"
6. **Confidence analysis**: "신뢰도 분석"
7. **Recheck info**: "재검사 현황"
8. **YOLO stats**: "YOLO 탐지 성능"
9. **Pipeline**: "파이프라인 설명"

Please ask a specific question about the analysis results!"""

    return response


# ============================================================
# Main Gradio Interface
# ============================================================

def create_interface():
    """Create the main Gradio interface."""

    custom_css = """
    .gradio-container { max-width: 1400px !important; }
    .tab-nav button { font-size: 16px !important; padding: 12px 24px !important; }
    """

    with gr.Blocks(title="dAIso - Semiconductor Defect Detection") as demo:

        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3a8a, #3b82f6); border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 32px;">dAIso</h1>
            <p style="color: #bfdbfe; margin: 10px 0 0 0; font-size: 16px;">
                Semiconductor Defect Detection System | Agent V10: YOLO + Two-Stage GPT-4o
            </p>
        </div>
        """)

        with gr.Tabs():
            # Tab 1: Dashboard
            with gr.TabItem("Dashboard", id=1):
                gr.HTML(get_dashboard_stats())

            # Tab 2: Individual Results
            with gr.TabItem("Inspection Results", id=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_dropdown = gr.Dropdown(
                            choices=get_image_list(),
                            value="DEV_000",
                            label="Select Image",
                            interactive=True
                        )
                        image_display = gr.Image(
                            label="Sample Image",
                            type="filepath",
                            height=300
                        )
                    with gr.Column(scale=2):
                        result_html = gr.HTML()

                def update_result(image_id):
                    html, img_url = format_result_detail(image_id)
                    return html, img_url

                image_dropdown.change(
                    fn=update_result,
                    inputs=[image_dropdown],
                    outputs=[result_html, image_display]
                )

                # Initialize with first result
                demo.load(
                    fn=lambda: update_result("DEV_000"),
                    outputs=[result_html, image_display]
                )

            # Tab 3: Visualizations
            with gr.TabItem("Visualizations", id=3):
                gr.Markdown("### Classification & Defect Distribution")
                chart1 = gr.Image(label="Classification Charts", type="filepath")

                gr.Markdown("### Confidence & Processing Time Analysis")
                chart2 = gr.Image(label="Performance Charts", type="filepath")

                gr.Markdown("### YOLO Detection Performance")
                chart3 = gr.Image(label="YOLO Performance Charts", type="filepath")

                refresh_btn = gr.Button("Refresh Charts", variant="primary")

                def generate_charts():
                    import tempfile

                    # Save charts to temp files
                    chart1_buf = create_classification_chart()
                    chart2_buf = create_confidence_chart()
                    chart3_buf = create_yolo_performance_chart()

                    # Save to temp files
                    chart1_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                    chart2_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                    chart3_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name

                    with open(chart1_path, 'wb') as f:
                        f.write(chart1_buf.read())
                    with open(chart2_path, 'wb') as f:
                        f.write(chart2_buf.read())
                    with open(chart3_path, 'wb') as f:
                        f.write(chart3_buf.read())

                    return chart1_path, chart2_path, chart3_path

                refresh_btn.click(
                    fn=generate_charts,
                    outputs=[chart1, chart2, chart3]
                )

                demo.load(
                    fn=generate_charts,
                    outputs=[chart1, chart2, chart3]
                )

            # Tab 4: Chatbot
            with gr.TabItem("AI Assistant", id=4):
                gr.Markdown("""
                ### Analysis Assistant
                Ask questions about the inspection results in Korean or English.

                **Example questions:**
                - "전체 분석 요약해줘"
                - "어떤 결함이 가장 많이 발견됐어?"
                - "DEV_005 결과 알려줘"
                - "신뢰도가 낮은 샘플은?"
                - "파이프라인 설명해줘"
                """)

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Enter your question here...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                clear_btn = gr.Button("Clear Chat")

                def respond(message, history):
                    if not message.strip():
                        return history, ""

                    bot_response = get_chatbot_response(message, history)
                    history = history + [(message, bot_response)]
                    return history, ""

                send_btn.click(
                    fn=respond,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input]
                )

                msg_input.submit(
                    fn=respond,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input]
                )

                clear_btn.click(
                    fn=lambda: [],
                    outputs=[chatbot]
                )

            # Tab 5: About
            with gr.TabItem("About", id=5):
                gr.Markdown("""
                ## Agent V10 - Semiconductor Defect Detection System

                ### Pipeline Architecture

                ```
                Image Input
                    │
                    ▼
                ┌─────────────────────┐
                │  Stage 0: YOLO      │  → Component Detection (holes, leads, body)
                │  (Roboflow API)     │  → Bounding boxes & confidence scores
                └─────────────────────┘
                    │
                    ▼
                ┌─────────────────────┐
                │  Stage 1: GPT-4o    │  → Visual Observation with YOLO hints
                │  (Observation)      │  → Body/Lead/Arrangement analysis
                └─────────────────────┘
                    │
                    ▼
                ┌─────────────────────┐
                │  Stage 2: GPT-4o    │  → Defect scoring system
                │  (Decision)         │  → Critical/High/Medium rules
                └─────────────────────┘
                    │
                    ▼
                ┌─────────────────────┐
                │  Conditional        │  → Confidence-triggered verification
                │  Recheck            │  → 4 types of focus methods
                └─────────────────────┘
                    │
                    ▼
                ┌─────────────────────┐
                │  Weighted Voting    │  → Final classification
                │  System             │  → Normal (0) or Abnormal (1)
                └─────────────────────┘
                ```

                ### Defect Classification Rules

                | Category | Defects | Score |
                |----------|---------|-------|
                | **Critical** | missed_hole, floating_lead, crossed/tangled | Auto-fail |
                | **High** | severe_tilt, blob_shape, deformation | +3~4 |
                | **Medium** | asymmetry, minor_tilt | +2 |

                **Classification:** Score ≥ 3 → Abnormal (1), Score < 3 → Normal (0)

                ### Technology Stack

                - **LLM**: GPT-4o (via Luxia Cloud Bridge API)
                - **Object Detection**: Roboflow Workflow API
                - **Tracing**: LangSmith
                - **Image Processing**: OpenCV, Pillow
                - **UI**: Gradio

                ---

                **Team**: Dong-Hyeon Lim, Munkyeong Suh
                **Version**: V10 (YOLO + Two-Stage GPT-4o)
                """)

        gr.HTML("""
        <div style="text-align: center; padding: 15px; margin-top: 20px; color: #64748b; font-size: 13px;">
            Hyundai NGV AI Agent Hackathon 2026 | Powered by GPT-4o & Roboflow
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
