#!/usr/bin/env python3
"""Demo scenario scripts for Delhi Scheme Saathi.

Three personas demonstrating the bot's capabilities:
1. Sunita - Widow seeking pension (Hindi, voice-first)
2. Rajesh - Unemployed person starting business (Hinglish)
3. Priya - New mother seeking housing (Bilingual)

Run with: python scripts/demo_scenarios.py [scenario_name]
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DemoMessage:
    """A single message in a demo scenario."""
    role: str  # "user" or "assistant"
    text: str
    notes: str = ""  # Notes for demo presenter


@dataclass
class DemoScenario:
    """A complete demo scenario with multiple turns."""
    name: str
    persona: str
    description: str
    life_event: str
    language: str
    messages: list[DemoMessage]
    key_features: list[str]
    duration_seconds: int


# =============================================================================
# Scenario 1: Sunita - Widow Pension (Hindi)
# =============================================================================

SUNITA_SCENARIO = DemoScenario(
    name="sunita_widow",
    persona="Sunita - 58 year old widow",
    description="Hindi-speaking widow seeking pension after husband's death",
    life_event="DEATH_IN_FAMILY",
    language="hi",
    duration_seconds=90,
    key_features=[
        "Life event detection (widowhood)",
        "Empathetic response in Hindi",
        "Profile extraction from conversation",
        "Widow Pension scheme match",
        "Document procurement guidance",
        "REJECTION PREVENTION (key differentiator)",
        "Nearest CSC office",
    ],
    messages=[
        DemoMessage(
            role="user",
            text="Namaste, mera pati guzar gaye",
            notes="Voice input in Hindi - triggers DEATH_IN_FAMILY classification"
        ),
        DemoMessage(
            role="assistant",
            text="🙏 नमस्ते बहन जी। आपके पति के निधन का मुझे बहुत दुख है। इस कठिन समय में मैं आपकी मदद करना चाहता हूं। विधवा पेंशन और अन्य सहायता योजनाओं के लिए, क्या आप अपनी उम्र बता सकती हैं?",
            notes="Empathetic acknowledgment + profile collection starts"
        ),
        DemoMessage(
            role="user",
            text="Main 58 saal ki hoon, SC category, income 80000 per year",
            notes="Profile info provided in natural conversation"
        ),
        DemoMessage(
            role="assistant",
            text="""✅ आपकी जानकारी मिल गई।

🙏 *विधवा पेंशन योजना (Widow Pension)*
💰 लाभ: ₹2,500 प्रति माह
✅ आप पात्र हैं: आयु 58 ✓ आय ₹80K ✓ SC ✓

क्या आप इस योजना के बारे में विस्तार से जानना चाहेंगी?""",
            notes="Scheme matched with eligibility explanation"
        ),
        DemoMessage(
            role="user",
            text="Haan, mujhe documents ki jankari chahiye",
            notes="User asks for document guidance"
        ),
        DemoMessage(
            role="assistant",
            text="""📄 *आवश्यक दस्तावेज:*

📌 *पहले ये बनवाएं:*
  1. आधार कार्ड (मुफ्त, UIDAI)
  2. मृत्यु प्रमाण पत्र (₹10, MCD/Hospital)

📄 *फिर बनवाएं:* आय प्रमाण पत्र
🏛 कहाँ से: SDM Office
💲 शुल्क: ₹10 (BPL: मुफ्त)
⏱ समय: 7-15 दिन

⚠️ *ध्यान रखें:* नाम आधार से बिल्कुल मैच होना चाहिए""",
            notes="Document chain with prerequisites shown"
        ),
        DemoMessage(
            role="user",
            text="Application reject hone se kaise bachun?",
            notes="User asks about rejection prevention - KEY MOMENT"
        ),
        DemoMessage(
            role="assistant",
            text="""⚠️ *अस्वीकृति से बचें:*

🔴 *गंभीर:* मृत्यु प्रमाण पत्र में नाम मिसमैच
   ✅ बचाव: मृत्यु प्रमाण पत्र में पति का नाम आधार से मैच करवाएं

🔴 *गंभीर:* आय प्रमाण पत्र में overstatement
   ✅ बचाव: वास्तविक आय ही दर्ज करवाएं (₹1 लाख से कम रहे)

🟠 *सावधान:* बैंक खाता NPCI से लिंक नहीं
   ✅ बचाव: Aadhaar-enabled bank account होना चाहिए

क्या आपको नजदीकी केंद्र की जानकारी चाहिए?""",
            notes="REJECTION PREVENTION shown with severity levels - KEY DIFFERENTIATOR"
        ),
        DemoMessage(
            role="user",
            text="Haan, kahan jaana hoga?",
            notes="User asks for nearest office"
        ),
        DemoMessage(
            role="assistant",
            text="""🏛️ *नजदीकी केंद्र:*

📍 *SDM Office - Central Delhi*
🚶 दूरी: 2.8 km
📫 Minto Road, New Delhi
📞 011-23237655
🕐 सोम-शनि 9:30 AM - 5:00 PM

📌 साथ ले जाएं: आधार, मृत्यु प्रमाण पत्र, पासपोर्ट फोटो, बैंक पासबुक

कोई और सवाल हो तो बताएं। 🙏""",
            notes="Nearest CSC with all details"
        ),
    ]
)


# =============================================================================
# Scenario 2: Rajesh - Self-Employment (Hinglish)
# =============================================================================

RAJESH_SCENARIO = DemoScenario(
    name="rajesh_business",
    persona="Rajesh - 28 year old unemployed youth",
    description="Hinglish-speaking young man who lost job, wants to start business",
    life_event="JOB_LOSS",
    language="hinglish",
    duration_seconds=60,
    key_features=[
        "Life event discovery (JOB_LOSS + BUSINESS_STARTUP)",
        "Hinglish language support",
        "RGSRY loan scheme match",
        "Document dependency chain",
    ],
    messages=[
        DemoMessage(
            role="user",
            text="Job chali gayi hai, kuch business karna hai",
            notes="Dual life event: JOB_LOSS + BUSINESS_STARTUP"
        ),
        DemoMessage(
            role="assistant",
            text="Arey Rajesh bhai, job jaana mushkil hota hai, par business shuru karna ek accha decision hai! 💪 Delhi mein self-employment ke liye acchi schemes hain. Pehle thoda batao - aapki age kitni hai aur kya category hai?",
            notes="Empathetic + encouraging Hinglish response"
        ),
        DemoMessage(
            role="user",
            text="28 saal, OBC, abhi unemployed hoon",
            notes="Profile info"
        ),
        DemoMessage(
            role="assistant",
            text="""Great! Aapke liye yeh scheme perfect hai:

💼 *Rajiv Gandhi Swavlamban Rojgar Yojana (RGSRY)*
💰 Loan: ₹7.5 लाख तक (no collateral)
📊 Subsidy: 25-35%
✅ आप eligible हैं: Age 28 ✓ OBC ✓ Unemployed ✓

Is scheme ke baare mein detail chahiye?""",
            notes="RGSRY scheme matched"
        ),
        DemoMessage(
            role="user",
            text="Haan, documents kya lagenge?",
            notes="Document query"
        ),
        DemoMessage(
            role="assistant",
            text="""📄 *Documents for RGSRY:*

📌 *Pehle yeh banwao:*
  1. Aadhaar Card (free)
  2. Domicile Certificate (₹10)
  3. OBC Certificate (₹10)

📄 *Phir apply karo:*
🏛 Kahan: District Industries Centre (DIC)
🌐 Online: dcmsme.gov.in
⏱ Processing: 30-45 days

⚠️ *Dhyan do:* Business plan ready rakhna - 2-page summary kaafi hai!""",
            notes="Document chain with business plan tip"
        ),
    ]
)


# =============================================================================
# Scenario 3: Priya - Housing (Bilingual)
# =============================================================================

PRIYA_SCENARIO = DemoScenario(
    name="priya_housing",
    persona="Priya - 32 year old expecting mother",
    description="Bilingual professional seeking housing under PMAY",
    life_event="HOUSING",
    language="bilingual",
    duration_seconds=60,
    key_features=[
        "Bilingual interaction (Hindi + English)",
        "PMAY-U housing scheme",
        "Application walkthrough",
    ],
    messages=[
        DemoMessage(
            role="user",
            text="Hi, I'm looking for housing schemes. Mujhe apna ghar chahiye.",
            notes="Bilingual query - HOUSING life event"
        ),
        DemoMessage(
            role="assistant",
            text="Hello! 🏠 Apna ghar - that's a great goal! Delhi mein housing ke liye Pradhan Mantri Awas Yojana (PMAY) available hai. Eligibility check karne ke liye, can you tell me your annual family income?",
            notes="Bilingual response matching user"
        ),
        DemoMessage(
            role="user",
            text="Around 5 lakh per year, we are MIG-I category",
            notes="Profile info"
        ),
        DemoMessage(
            role="assistant",
            text="""Perfect! PMAY-U aapke liye eligible hai:

🏠 *Pradhan Mantri Awas Yojana - Urban 2.0*
💰 Benefit: ₹2.5 lakh subsidy on home loan
📋 Category: MIG-I (₹6-12 lakh income) ✓
🏛 Ministry: Housing & Urban Affairs

Application process:
1. Online: pmaymis.gov.in
2. Through bank (SBI, HDFC, etc.)
3. CSC center

Kaunsa option prefer karoge?""",
            notes="PMAY scheme with application options"
        ),
        DemoMessage(
            role="user",
            text="Online kaise apply karun?",
            notes="Application help"
        ),
        DemoMessage(
            role="assistant",
            text="""🌐 *Online Application Steps:*

1. Visit pmaymis.gov.in
2. Click "Citizen Assessment" → "Benefit under other 3 components"
3. Enter Aadhaar number for verification
4. Fill family details & income info
5. Upload documents:
   • Aadhaar (self + spouse)
   • Income certificate
   • Property documents (if buying)
6. Submit & note application ID

⏱ Processing: 30-60 days for approval
💡 Tip: Apply through your bank for faster processing!

Koi aur question?""",
            notes="Step-by-step application guidance"
        ),
    ]
)


# =============================================================================
# Demo Runner
# =============================================================================

SCENARIOS = {
    "sunita": SUNITA_SCENARIO,
    "rajesh": RAJESH_SCENARIO,
    "priya": PRIYA_SCENARIO,
}


def print_scenario(scenario: DemoScenario, interactive: bool = False):
    """Print a demo scenario for presentation."""
    print("\n" + "=" * 70)
    print(f"DEMO SCENARIO: {scenario.name.upper()}")
    print("=" * 70)
    print(f"\n📋 Persona: {scenario.persona}")
    print(f"📝 Description: {scenario.description}")
    print(f"🎯 Life Event: {scenario.life_event}")
    print(f"🗣️ Language: {scenario.language}")
    print(f"⏱️ Duration: ~{scenario.duration_seconds} seconds")
    print(f"\n🔑 Key Features Demonstrated:")
    for feature in scenario.key_features:
        print(f"   • {feature}")

    print("\n" + "-" * 70)
    print("CONVERSATION FLOW")
    print("-" * 70)

    for i, msg in enumerate(scenario.messages, 1):
        if interactive:
            input("\n[Press Enter for next message...]")

        role_icon = "👤" if msg.role == "user" else "🤖"
        role_label = "USER" if msg.role == "user" else "BOT"

        print(f"\n{role_icon} [{role_label}]")
        if msg.notes:
            print(f"   📌 Note: {msg.notes}")
        print(f"\n{msg.text}")

    print("\n" + "=" * 70)
    print("END OF SCENARIO")
    print("=" * 70 + "\n")


async def run_live_demo(scenario: DemoScenario):
    """Run scenario against live API."""
    from dotenv import load_dotenv
    load_dotenv()

    import httpx

    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    user_id = f"demo_{scenario.name}"

    print(f"\n🚀 Running live demo against {base_url}")
    print(f"   User ID: {user_id}\n")

    async with httpx.AsyncClient() as client:
        for msg in scenario.messages:
            if msg.role == "user":
                print(f"👤 USER: {msg.text}")

                response = await client.post(
                    f"{base_url}/api/chat",
                    json={"user_id": user_id, "message": msg.text},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    print(f"🤖 BOT: {data.get('response', 'No response')}\n")
                else:
                    print(f"❌ Error: {response.status_code} - {response.text}\n")

                await asyncio.sleep(1)


def export_scenario(scenario: DemoScenario, format: str = "json"):
    """Export scenario for documentation or video script."""
    if format == "json":
        data = {
            "name": scenario.name,
            "persona": scenario.persona,
            "description": scenario.description,
            "life_event": scenario.life_event,
            "language": scenario.language,
            "duration_seconds": scenario.duration_seconds,
            "key_features": scenario.key_features,
            "messages": [
                {"role": m.role, "text": m.text, "notes": m.notes}
                for m in scenario.messages
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    elif format == "markdown":
        lines = [
            f"# Demo Scenario: {scenario.persona}",
            "",
            f"**Duration:** {scenario.duration_seconds} seconds",
            f"**Life Event:** {scenario.life_event}",
            "",
            "## Key Features",
            "",
        ]
        for feature in scenario.key_features:
            lines.append(f"- {feature}")

        lines.extend(["", "## Conversation Flow", ""])

        for msg in scenario.messages:
            role = "**User:**" if msg.role == "user" else "**Bot:**"
            lines.append(f"{role}")
            if msg.notes:
                lines.append(f"> _{msg.notes}_")
            lines.append("")
            lines.append(msg.text)
            lines.append("")

        return "\n".join(lines)

    return ""


def main():
    parser = argparse.ArgumentParser(description="Delhi Scheme Saathi Demo Scenarios")
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode (press Enter for each message)",
    )
    parser.add_argument(
        "--live", "-l",
        action="store_true",
        help="Run against live API",
    )
    parser.add_argument(
        "--export", "-e",
        choices=["json", "markdown"],
        help="Export scenario to format",
    )

    args = parser.parse_args()

    if args.scenario == "all":
        scenarios_to_run = list(SCENARIOS.values())
    else:
        scenarios_to_run = [SCENARIOS[args.scenario]]

    for scenario in scenarios_to_run:
        if args.export:
            output = export_scenario(scenario, args.export)
            print(output)
        elif args.live:
            asyncio.run(run_live_demo(scenario))
        else:
            print_scenario(scenario, args.interactive)


if __name__ == "__main__":
    main()
