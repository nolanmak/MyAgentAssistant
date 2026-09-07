//! Block-kit-ish layout for approval messages.
//!
//! Serenity's model builders are verbose; we build simple component collections
//! here so the broker stays readable.

use augmentagent_store::Email;
use serenity::all::{
    ActionRowComponent, ButtonStyle, CreateActionRow, CreateButton, CreateEmbed, CreateEmbedFooter,
    CreateInputText, CreateModal, CreateMessage, CreateSelectMenu, CreateSelectMenuKind,
    CreateSelectMenuOption, InputTextStyle,
};

use crate::custom_id::{CustomId, Verb};
use crate::presets::{MAX_REDRAFT_ITERATIONS, PRESETS};

const MAX_EMBED_DESCRIPTION: usize = 3800;
const SEPARATOR: &str = "\n\n— DRAFT —\n\n";

/// Sentinel that fences the "needs your input" payload appended to the
/// persisted draft body by the email channel (#35 Phase 5). It is an HTML
/// comment so it never reaches a recipient even if a code path skipped the
/// strip, and its presence is the ONLY trigger for the card's needs-input
/// field + button. The off / shadow ask-resolve paths never emit it, so a
/// draft with no marker renders byte-identically to pre-#35.
///
/// Body format (one ask per line, between the open/close fences):
/// ```text
/// <!--aa:needs-input
/// share_doc|the Q4 board deck
/// intro|Sarah Chen
/// -->
/// ```
const NEEDS_INPUT_OPEN: &str = "<!--aa:needs-input\n";
const NEEDS_INPUT_CLOSE: &str = "\n-->";

/// One ask the resolver could not auto-fill, decoded from the draft marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeedsInput {
    /// Resolver-kind tag (`scheduling`, `share_doc`, …). Opaque to the card;
    /// only used to label the field row and key the modal.
    pub kind: String,
    /// The ask paraphrase, shown verbatim so the user knows what to supply.
    pub text: String,
}

/// Append the needs-input marker to a draft body for persistence. Called by
/// the email channel ONLY when live ask-resolve produced unresolved asks;
/// `asks` empty ⇒ the draft is returned unchanged (byte-identical path).
/// `(kind, text)` pairs are sanitized so the marker stays single-block and
/// round-trips through [`split_needs_input`].
pub fn append_needs_input_marker(draft: &str, asks: &[(String, String)]) -> String {
    if asks.is_empty() {
        return draft.to_string();
    }
    let mut out = String::with_capacity(draft.len() + 64 + asks.len() * 48);
    out.push_str(draft.trim_end());
    out.push_str("\n\n");
    out.push_str(NEEDS_INPUT_OPEN);
    for (kind, text) in asks {
        // `|` is the field delimiter and newlines end a record — strip both
        // from the payload so a pathological ask text can't break the frame.
        let k = sanitize_marker_field(kind);
        let t = sanitize_marker_field(text);
        out.push_str(&k);
        out.push('|');
        out.push_str(&t);
        out.push('\n');
    }
    // Trim the trailing record newline so CLOSE's leading `\n` is the only one.
    out.pop();
    out.push_str(NEEDS_INPUT_CLOSE);
    out
}

fn sanitize_marker_field(s: &str) -> String {
    s.replace(['|', '\n', '\r'], " ").trim().to_string()
}

/// Sentinel fencing the "assumed facts" payload the drafter appends when its
/// reply rests on material facts neither the thread nor the resolved context
/// established (#785). Same carrier discipline as the needs-input marker: an
/// HTML comment so it can never reach a recipient, card-only, and absent by
/// default — a draft with no fence renders byte-identically to pre-#785.
///
/// Body format (one assumed fact per line, between the fences):
/// ```text
/// <!--aa:assumes
/// you're free on the 14th - not verified against calendar
/// -->
/// ```
///
/// The lines are model-authored, so an inbound email could in principle steer
/// the drafter into emitting a bogus one. That is tolerable: the payload is a
/// read-only caution on the approval card and drives no action, so the worst
/// case is a warning the owner dismisses.
const ASSUMES_OPEN: &str = "<!--aa:assumes\n";
const ASSUMES_CLOSE: &str = "\n-->";
/// [`ASSUMES_OPEN`] without its newline — lets [`strip_assumes_for_send`]
/// catch a fence the model opened but never closed.
const ASSUMES_OPEN_PREFIX: &str = "<!--aa:assumes";

/// Cap on assumption lines carried by a card. Mirrors the limit stated in the
/// draft prompt so a runaway model can't flood the embed field.
const MAX_ASSUMES: usize = 5;

/// Append the assumed-facts marker to a draft body. `facts` empty (or all
/// blank) ⇒ the draft is returned unchanged, keeping the no-assumptions path
/// byte-identical. Newlines are stripped from each fact so the frame stays a
/// single block and round-trips through [`split_assumes`].
pub fn append_assumes_marker(draft: &str, facts: &[String]) -> String {
    let lines: Vec<String> = facts
        .iter()
        .map(|f| f.replace(['\n', '\r'], " ").trim().to_string())
        .filter(|f| !f.is_empty())
        .take(MAX_ASSUMES)
        .collect();
    if lines.is_empty() {
        return draft.to_string();
    }
    let mut out = String::with_capacity(draft.len() + 32 + lines.len() * 48);
    out.push_str(draft.trim_end());
    out.push_str("\n\n");
    out.push_str(ASSUMES_OPEN);
    out.push_str(&lines.join("\n"));
    out.push_str(ASSUMES_CLOSE);
    out
}

/// Split a persisted draft into the human-facing draft and the assumed facts
/// carried in its marker. No marker ⇒ `(draft, vec![])`. Tolerant: a malformed
/// marker is treated as absent, so a half-parsed HTML comment is never
/// rendered as a warning.
///
/// Unlike [`split_needs_input`], this SPLICES the fence out instead of
/// truncating at it. The fence can legitimately have text after it — the #629
/// `[to:]`/`[cc:]` envelope display markers and the needs-input marker are
/// both appended later — and truncating would silently drop them from the card.
pub fn split_assumes(draft: &str) -> (String, Vec<String>) {
    let Some(open_at) = draft.rfind(ASSUMES_OPEN) else {
        return (draft.to_string(), Vec::new());
    };
    let after_open = open_at + ASSUMES_OPEN.len();
    let Some(rel_close) = draft[after_open..].find(ASSUMES_CLOSE) else {
        return (draft.to_string(), Vec::new());
    };
    let facts: Vec<String> = draft[after_open..after_open + rel_close]
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .take(MAX_ASSUMES)
        .collect();
    if facts.is_empty() {
        return (draft.to_string(), Vec::new());
    }
    let before = draft[..open_at].trim_end();
    let after = draft[after_open + rel_close + ASSUMES_CLOSE.len()..].trim_start();
    let human = if after.is_empty() {
        before.to_string()
    } else {
        format!("{before}\n\n{after}")
    };
    (human, facts)
}

/// Scrub every trace of the assumes marker from a body bound for a real
/// recipient. Stricter than [`split_assumes`], which deliberately leaves a
/// malformed fence alone so the card never shows half-parsed markup: a Gmail
/// body has no such luxury, so an unclosed `<!--aa:assumes` is truncated away.
pub fn strip_assumes_for_send(body: &str) -> String {
    let (clean, _) = split_assumes(body);
    match clean.rfind(ASSUMES_OPEN_PREFIX) {
        Some(at) => clean[..at].trim_end().to_string(),
        None => clean,
    }
}

/// Split a persisted draft into the human-facing draft and any needs-input
/// asks carried in the trailing marker. No marker ⇒ `(draft, vec![])`, so
/// every legacy / off-path draft is unaffected. Tolerant: a malformed marker
/// is treated as absent (the raw text is returned, never shown half-parsed).
pub fn split_needs_input(draft: &str) -> (String, Vec<NeedsInput>) {
    let Some(open_at) = draft.rfind(NEEDS_INPUT_OPEN) else {
        return (draft.to_string(), Vec::new());
    };
    let after_open = open_at + NEEDS_INPUT_OPEN.len();
    let Some(rel_close) = draft[after_open..].find(NEEDS_INPUT_CLOSE) else {
        return (draft.to_string(), Vec::new());
    };
    let body = &draft[after_open..after_open + rel_close];
    let mut asks = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((kind, text)) = line.split_once('|') {
            let kind = kind.trim();
            let text = text.trim();
            if !kind.is_empty() && !text.is_empty() {
                asks.push(NeedsInput {
                    kind: kind.to_string(),
                    text: text.to_string(),
                });
            }
        }
    }
    if asks.is_empty() {
        return (draft.to_string(), Vec::new());
    }
    let human = draft[..open_at].trim_end().to_string();
    (human, asks)
}

/// Render the user's supplied values as a structured Revise-feedback string.
/// Reuses the existing redraft path: the drafter re-writes the reply with the
/// concrete values substituted, exactly as if a resolver had filled them. The
/// `<resolved_asks>`-style framing keeps the anti-placeholder contract.
pub fn fill_feedback(filled: &[(NeedsInput, String)]) -> String {
    let mut s = String::from(
        "The user has supplied the previously-missing concrete values below. \
         Rewrite the draft so each is used EXACTLY as given. Do NOT write a \
         placeholder and do NOT say \"I'll send it shortly\". Keep everything \
         else about the draft the same.\n",
    );
    for (ask, value) in filled {
        s.push_str(&format!(
            "- {} (for \"{}\"): {}\n",
            label_for(&ask.kind),
            ask.text,
            value.trim()
        ));
    }
    s
}

/// Human label for a resolver-kind tag. Mirrors
/// `augmentagent_channel_core::ResolverKind::label` but kept local so the
/// approval crate doesn't take a dependency on channel-core (the dep edge
/// runs the other way).
fn label_for(kind: &str) -> &'static str {
    match kind {
        "scheduling" => "Proposed meeting time",
        "calendly" => "Booking link",
        "meeting_link" => "Video-call link",
        "share_doc" => "Document link",
        "intro" => "Introduction",
        _ => "Detail",
    }
}

/// Plain-text "heads up" card for triage-flagged emails. No buttons, no draft.
pub fn flag_notice_message(email: &Email, reason: &str) -> CreateMessage {
    let subject = truncate(&email.subject, 256);
    let from = truncate(&email.from, 200);
    let reason = truncate(reason, 500);
    let content = format!(
        "🚩 **Important** — from `{from}`\n**{subject}**\n_reason: {reason}_"
    );
    CreateMessage::new().content(content)
}

/// Build an approval card. `redraft_count` is how many times this draft has
/// already been refined (0 on first post). When the count is under
/// [`MAX_REDRAFT_ITERATIONS`] a second action row — the "Quick refine…"
/// `StringSelect` (#34) — is attached so presets can stack. At/above the cap
/// the menu is dropped (the footer says so) and only Approve/Revise/Skip
/// remain, forcing a terminal action or free-form Revise.
pub fn approval_message(
    action_id: &str,
    email: &Email,
    draft: &str,
    redraft_count: i64,
) -> CreateMessage {
    // Strip the needs-input marker BEFORE anything else so the human never
    // sees the raw `<!--aa:needs-input …-->` fence and the draft preview is
    // the real reply text. No marker ⇒ `human == draft`, byte-identical card.
    let (human_draft, needs) = split_needs_input(draft);
    // #785 — then lift out the assumed facts, so the preview is the reply the
    // recipient would get and the assumptions get their own field.
    let (human_draft, assumes) = split_assumes(&human_draft);
    let at_cap = redraft_count >= MAX_REDRAFT_ITERATIONS;
    let footer = if redraft_count == 0 {
        "AugmentAgent approval".to_string()
    } else if at_cap {
        format!(
            "AugmentAgent approval · draft v{} · refine cap reached — Approve/Skip or use Revise",
            redraft_count + 1
        )
    } else {
        format!("AugmentAgent approval · draft v{}", redraft_count + 1)
    };
    let mut embed = CreateEmbed::new()
        .title(truncate(&email.subject, 256))
        .description(format_body(&email.body, &human_draft))
        .field("From", truncate(&email.from, 256), true)
        .field("MessageId", truncate(&email.message_id, 256), true);
    if !needs.is_empty() {
        embed = embed.field(
            "⚠️ Needs your input",
            truncate(&format_needs_input(&needs), 1024),
            false,
        );
    }
    if !assumes.is_empty() {
        embed = embed.field("⚠ Assumes", truncate(&format_assumes(&assumes), 1024), false);
    }
    let embed = embed.footer(CreateEmbedFooter::new(footer));

    // #927 — a merge card carries evidence, not a draft: nothing to redraft,
    // so Approve (which runs the merge) and Skip are the only verbs offered.
    if email.kind == "identity_merge" {
        let row = CreateActionRow::Buttons(vec![
            CreateButton::new(CustomId::new(action_id, Verb::Approve).to_string())
                .label("Approve & Merge")
                .style(ButtonStyle::Success),
            CreateButton::new(CustomId::new(action_id, Verb::Skip).to_string())
                .label("Skip")
                .style(ButtonStyle::Secondary),
        ]);
        return CreateMessage::new().embed(embed).components(vec![row]);
    }

    let button_row = CreateActionRow::Buttons(vec![
        CreateButton::new(CustomId::new(action_id, Verb::Approve).to_string())
            .label("Approve & Send")
            .style(ButtonStyle::Success),
        CreateButton::new(CustomId::new(action_id, Verb::Revise).to_string())
            .label("Revise")
            .style(ButtonStyle::Primary),
        CreateButton::new(CustomId::new(action_id, Verb::Skip).to_string())
            .label("Skip")
            .style(ButtonStyle::Secondary),
    ]);

    let mut rows = vec![button_row];
    // Row 2: the "Provide missing info" button — ONLY when the draft carries
    // unresolved asks. Sits on its own row so the existing quick-refine
    // StringSelect (also row-2-or-later) and the unchanged row-1 buttons are
    // untouched.
    if !needs.is_empty() {
        rows.push(CreateActionRow::Buttons(vec![CreateButton::new(
            CustomId::new(action_id, Verb::FillAsk).to_string(),
        )
        .label("Provide missing info")
        .style(ButtonStyle::Danger)]));
    }
    if !at_cap {
        rows.push(quick_refine_row(action_id));
    }
    // #501 — the Schedule select, last and independent of the `!at_cap`
    // gate above: an at-cap card must still be schedulable. Offered only on
    // cards the scheduled pipeline can actually fire (a deferred Gmail
    // send_draft) — the handler refuses non-email schedules too, but the
    // control shouldn't be rendered where it can only fail (#501 review).
    // Worst case is 4 of Discord's 5 rows (buttons + FillAsk + QuickRefine +
    // Schedule) — comfortably inside the row budget.
    if is_schedulable(email) {
        rows.push(schedule_row(action_id));
    }
    CreateMessage::new().embed(embed).components(rows)
}

/// #501 — can this card's action ride the scheduled-send pipeline? Only the
/// Gmail fallthrough path in `run_approve` can: platform `gmail` and not a
/// LinkedIn-entity row (LinkedIn reuses platform `gmail` on legacy rows and
/// is identified by its `linkedin:` account-entity prefix — kept as a local
/// literal because the dep edge to `augmentagent-channel-linkedin` runs the
/// other way, mirroring `ACCOUNT_PREFIX` there).
fn is_schedulable(email: &Email) -> bool {
    email.platform == "gmail"
        && !email
            .account_entity_id
            .as_deref()
            .unwrap_or("")
            .starts_with("linkedin:")
}

/// One bullet per unresolved ask for the card field.
fn format_needs_input(needs: &[NeedsInput]) -> String {
    let mut s = String::new();
    for n in needs {
        s.push_str("• **");
        s.push_str(label_for(&n.kind));
        s.push_str("** — ");
        s.push_str(n.text.trim());
        s.push('\n');
    }
    s.push_str("_Click **Provide missing info** to supply these; the draft re-renders with your values._");
    s
}

/// One bullet per assumed fact for the card field (#785). The caption is what
/// turns the list into an actionable choice: the approver either accepts the
/// assumption or hits Revise.
fn format_assumes(facts: &[String]) -> String {
    let mut s = String::new();
    for f in facts {
        s.push_str("• ");
        s.push_str(f);
        s.push('\n');
    }
    s.push_str("_Not verified — Revise if any of these is wrong._");
    s
}

/// Modal that collects a value for each unresolved ask. One paragraph input
/// per ask (Discord caps modals at 5 inputs — resolver kinds top out at 5, so
/// this is always within budget; extra asks beyond 5 are dropped from the
/// modal but stay listed on the card field).
pub fn fill_ask_modal(action_id: &str, needs: &[NeedsInput]) -> CreateModal {
    let inputs: Vec<CreateActionRow> = needs
        .iter()
        .take(5)
        .enumerate()
        .map(|(i, n)| {
            // custom_id of each input encodes the kind so the submit handler
            // can pair value→ask without relying on positional order.
            let input = CreateInputText::new(
                InputTextStyle::Paragraph,
                truncate(&format!("{} — {}", label_for(&n.kind), n.text), 45),
                format!("ask{i}:{}", n.kind),
            )
            .required(false)
            .placeholder("Paste the value, or leave blank to skip")
            .max_length(1000);
            CreateActionRow::InputText(input)
        })
        .collect();
    CreateModal::new(
        CustomId::new(action_id, Verb::FillAskModal).to_string(),
        "Provide missing info",
    )
    .components(inputs)
}

/// Pair submitted modal inputs back to their asks. Each input's `custom_id`
/// is `ask<i>:<kind>`; the value is the user's text. Blank values are
/// dropped (the user chose to skip that one). Returns `(NeedsInput, value)`
/// pairs ready for [`fill_feedback`].
pub fn extract_fill_values(
    rows: &[serenity::all::ActionRow],
    needs: &[NeedsInput],
) -> Vec<(NeedsInput, String)> {
    let mut out = Vec::new();
    for row in rows {
        for c in &row.components {
            if let ActionRowComponent::InputText(input) = c {
                let Some(val) = input.value.as_ref() else {
                    continue;
                };
                if val.trim().is_empty() {
                    continue;
                }
                // `ask<i>:<kind>` — recover the kind, then match by position
                // or kind back to the original ask for its text.
                let kind = input
                    .custom_id
                    .split_once(':')
                    .map(|(_, k)| k.to_string())
                    .unwrap_or_default();
                let idx = input
                    .custom_id
                    .strip_prefix("ask")
                    .and_then(|r| r.split_once(':'))
                    .and_then(|(i, _)| i.parse::<usize>().ok());
                let ask = idx
                    .and_then(|i| needs.get(i))
                    .filter(|a| a.kind == kind)
                    .or_else(|| needs.iter().find(|a| a.kind == kind))
                    .cloned();
                if let Some(ask) = ask {
                    out.push((ask, val.clone()));
                }
            }
        }
    }
    out
}

/// The "Quick refine…" `StringSelect` action row (#34). Each option's value is
/// a preset id; the handler maps it to a canned `redraft_message` feedback
/// string. Re-attached on every re-render so presets stack until the cap.
fn quick_refine_row(action_id: &str) -> CreateActionRow {
    let options: Vec<CreateSelectMenuOption> = PRESETS
        .iter()
        .map(|p| CreateSelectMenuOption::new(p.label, p.id))
        .collect();
    let menu = CreateSelectMenu::new(
        CustomId::new(action_id, Verb::QuickRefine).to_string(),
        CreateSelectMenuKind::String { options },
    )
    .placeholder("Quick refine… (no typing)")
    .min_values(1)
    .max_values(1);
    CreateActionRow::SelectMenu(menu)
}

/// Select value that opens the custom-time modal instead of resolving to a
/// concrete offset (#501). Kept as a named constant so the event handler's
/// branch and the option list below can't drift apart.
pub const SCHEDULE_CUSTOM_VALUE: &str = "custom";

/// The Schedule select's options: label → SYMBOLIC value (#501). Values are
/// resolved to epoch-ms at CLICK time by `timeparse::resolve_token`, never at
/// render time — cards sit for hours/days, and a render-time "in 1 hour"
/// would be stale by the time it's picked.
const SCHEDULE_OPTIONS: &[(&str, &str)] = &[
    ("In 1 hour", "in1h"),
    ("In 3 hours", "in3h"),
    ("Tonight 7pm", "tonight-1900"),
    ("Tomorrow 9am", "tomorrow-0900"),
    ("Tomorrow 2pm", "tomorrow-1400"),
    ("Next Monday 9am", "next-monday-0900"),
    ("Custom…", SCHEDULE_CUSTOM_VALUE),
];

/// The "Schedule send…" `StringSelect` action row (#501). Attached to EVERY
/// approval card (including at-cap re-renders — see `approval_message`).
fn schedule_row(action_id: &str) -> CreateActionRow {
    let options: Vec<CreateSelectMenuOption> = SCHEDULE_OPTIONS
        .iter()
        .map(|(label, value)| CreateSelectMenuOption::new(*label, *value))
        .collect();
    let menu = CreateSelectMenu::new(
        CustomId::new(action_id, Verb::SchedulePick).to_string(),
        CreateSelectMenuKind::String { options },
    )
    .placeholder("Schedule send…")
    .min_values(1)
    .max_values(1);
    CreateActionRow::SelectMenu(menu)
}

/// Compact scheduled-notice message (#501) — the card's replacement once a
/// schedule is armed. Deliberately NOT an approval card: it must never carry
/// Approve/Revise/Skip (the draft is already approved-for-later), only the
/// three schedule escape hatches. `to_display` is the actual send target the
/// caller resolved — the #473 envelope To when one was recorded, else the
/// card's From line — so an overridden routing (intro pattern) shows the
/// real recipient, not the reply-to.
pub fn scheduled_notice_message(
    action_id: &str,
    email: &Email,
    sends_at_local: &str,
    sends_at_ms: i64,
    to_display: &str,
) -> CreateMessage {
    let sends_at_unix = sends_at_ms.div_euclid(1_000);
    let embed = CreateEmbed::new()
        .title(format!("Scheduled: {}", truncate(&email.subject, 245)))
        .description("This email is armed and will send automatically.")
        .field("To", truncate(to_display, 256), true)
        .field(
            "Sends automatically",
            format!(
                "{sends_at_local}\n<t:{sends_at_unix}:F> · <t:{sends_at_unix}:R>"
            ),
            false,
        )
        .footer(CreateEmbedFooter::new("AugmentAgent scheduled send"));
    let buttons = CreateActionRow::Buttons(vec![
        CreateButton::new(CustomId::new(action_id, Verb::SendNow).to_string())
            .label("Send now instead")
            .style(ButtonStyle::Secondary),
        CreateButton::new(CustomId::new(action_id, Verb::BackToQueue).to_string())
            .label("Back to queue")
            .style(ButtonStyle::Secondary),
        CreateButton::new(CustomId::new(action_id, Verb::CancelSchedule).to_string())
            .label("Cancel")
            .style(ButtonStyle::Danger),
    ]);
    CreateMessage::new().embed(embed).components(vec![buttons])
}

/// Custom-time modal opened by the `custom` schedule token (#501). One short
/// text input; the placeholder doubles as the format cheat-sheet. Known
/// cosmetic artifact: dismissing the modal leaves the select displaying
/// "Custom…" — re-picking still works.
pub fn schedule_modal(action_id: &str) -> CreateModal {
    let input = CreateInputText::new(InputTextStyle::Short, "When?", "when")
        .required(true)
        .placeholder("tomorrow 9am / fri 14:30 / in 3h / 2026-09-01 09:00")
        .max_length(100);
    CreateModal::new(
        CustomId::new(action_id, Verb::ScheduleModal).to_string(),
        "Schedule send",
    )
    .components(vec![CreateActionRow::InputText(input)])
}

pub fn revise_modal(action_id: &str, previous_feedback: Option<&str>) -> CreateModal {
    let input = CreateInputText::new(InputTextStyle::Paragraph, "Revision feedback", "feedback")
        .required(true)
        .placeholder("What should change about the draft?")
        .max_length(1500)
        .value(previous_feedback.unwrap_or(""));

    CreateModal::new(
        CustomId::new(action_id, Verb::ReviseModal).to_string(),
        "Revise draft",
    )
    .components(vec![CreateActionRow::InputText(input)])
}

fn format_body(email_body: &str, draft: &str) -> String {
    let budget = MAX_EMBED_DESCRIPTION.saturating_sub(SEPARATOR.len());
    let half = budget / 2;
    let email_part = truncate(email_body, half);
    let draft_part = truncate(draft, budget - email_part.len());
    format!("{email_part}{SEPARATOR}{draft_part}")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max.saturating_sub(3);
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Extract the user-entered feedback from a submitted modal. Returns `None`
/// if the modal didn't contain a text input (should not happen with our layout).
pub fn extract_feedback(rows: &[serenity::all::ActionRow]) -> Option<String> {
    for row in rows {
        for c in &row.components {
            if let ActionRowComponent::InputText(input) = c {
                if let Some(v) = &input.value {
                    return Some(v.clone());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_respects_char_boundary() {
        let s = "aéb"; // é = 2 bytes, total 4 bytes
        let t = truncate(s, 3);
        assert!(t.is_char_boundary(t.len()));
    }

    #[test]
    fn format_body_fits_budget() {
        let big = "a".repeat(10_000);
        let out = format_body(&big, &big);
        assert!(out.len() <= MAX_EMBED_DESCRIPTION);
        assert!(out.contains(SEPARATOR));
    }

    // ---------------------------------------------------------------------
    // #35 Phase 5: needs-input marker round-trip + card UX.
    // ---------------------------------------------------------------------

    fn email() -> Email {
        Email {
            attachments: Vec::new(),
            to: String::new(),
            cc: String::new(),
            message_id: "m1".into(),
            thread_id: Some("t1".into()),
            from: "peer@example.com".into(),
            subject: "Re: hi".into(),
            body: "the inbound message".into(),
            date: "2026-05-18T00:00:00Z".into(),
            account_entity_id: Some("acc".into()),
            platform: "gmail".into(),
            kind: "dm".into(),
        }
    }

    fn json(m: &CreateMessage) -> String {
        serde_json::to_string(m).expect("CreateMessage serializes")
    }

    #[test]
    fn no_marker_means_no_needs_input_and_unchanged_draft() {
        let (human, needs) = split_needs_input("Hi there,\n\nThanks!\n");
        assert_eq!(human, "Hi there,\n\nThanks!\n");
        assert!(needs.is_empty());
    }

    #[test]
    fn append_then_split_round_trips() {
        let draft = "Hi,\n\nHappy to help.\n\nBest,\nN";
        let asks = vec![
            ("share_doc".to_string(), "the Q4 board deck".to_string()),
            ("intro".to_string(), "Sarah Chen".to_string()),
        ];
        let marked = append_needs_input_marker(draft, &asks);
        assert!(marked.contains(NEEDS_INPUT_OPEN));
        let (human, needs) = split_needs_input(&marked);
        assert_eq!(human, draft.trim_end());
        assert_eq!(needs.len(), 2);
        assert_eq!(needs[0].kind, "share_doc");
        assert_eq!(needs[0].text, "the Q4 board deck");
        assert_eq!(needs[1].kind, "intro");
        assert_eq!(needs[1].text, "Sarah Chen");
    }

    #[test]
    fn append_with_no_asks_is_byte_identical() {
        let draft = "Just a normal reply.";
        assert_eq!(append_needs_input_marker(draft, &[]), draft);
    }

    #[test]
    fn marker_payload_with_pipe_and_newline_is_sanitized() {
        let asks = vec![(
            "share_doc".to_string(),
            "the | deck\nwith newlines".to_string(),
        )];
        let marked = append_needs_input_marker("body", &asks);
        let (_human, needs) = split_needs_input(&marked);
        assert_eq!(needs.len(), 1);
        // No raw pipe/newline survived into the decoded text.
        assert!(!needs[0].text.contains('|'));
        assert!(!needs[0].text.contains('\n'));
        assert!(needs[0].text.contains("deck"));
    }

    #[test]
    fn malformed_marker_is_treated_as_absent() {
        // Open fence but no close fence ⇒ return raw, no asks.
        let broken = format!("body\n{NEEDS_INPUT_OPEN}share_doc|x");
        let (human, needs) = split_needs_input(&broken);
        assert_eq!(human, broken);
        assert!(needs.is_empty());
    }

    #[test]
    fn approval_card_without_marker_is_byte_identical_to_legacy() {
        // The whole off-path guarantee: a marker-free draft must produce the
        // exact same card bytes as before #35 (no extra field, no extra row).
        let plain = "Hello,\n\nSounds good — see you then.\n\nBest,\nN";
        let with = json(&approval_message("act-1", &email(), plain, 0));
        // No needs-input field, no fill button.
        assert!(!with.contains("Needs your input"));
        assert!(!with.contains("Provide missing info"));
        assert!(!with.contains("fill_ask"));
        // Row-1 verbs intact.
        assert!(with.contains("aa:act-1:approve"));
        assert!(with.contains("aa:act-1:revise"));
        assert!(with.contains("aa:act-1:skip"));
        // The draft text is shown verbatim (no marker stripping artifacts).
        assert!(with.contains("Sounds good"));
    }

    #[test]
    fn approval_card_with_marker_adds_field_and_button_only() {
        let marked = append_needs_input_marker(
            "Hi,\n\nI'll get you that.\n\nN",
            &[("share_doc".into(), "the pitch deck".into())],
        );
        let v = json(&approval_message("act-2", &email(), &marked, 0));
        // New field + new button present.
        assert!(v.contains("Needs your input"));
        assert!(v.contains("Provide missing info"));
        assert!(v.contains("aa:act-2:fill_ask"));
        assert!(v.contains("Document link"));
        assert!(v.contains("the pitch deck"));
        // Row-1 still exactly Approve/Revise/Skip.
        assert!(v.contains("aa:act-2:approve"));
        assert!(v.contains("aa:act-2:revise"));
        assert!(v.contains("aa:act-2:skip"));
        // The raw marker fence must NOT leak into the rendered card.
        assert!(!v.contains("aa:needs-input"));
    }

    // ---------------------------------------------------------------------
    // #785: assumed-facts marker round-trip + card annotation.
    // ---------------------------------------------------------------------

    #[test]
    fn approval_card_surfaces_assumed_facts_and_hides_the_fence() {
        let draft = "Hi,\n\nThe 14th works.\n\nBest,\nN\n\n<!--aa:assumes\nyou're free on the 14th - not verified against calendar\nthe kickoff is still scoped to Q4\n-->";
        let v = json(&approval_message("act-a1", &email(), draft, 0));
        assert!(v.contains("Assumes"), "missing assumes field: {v}");
        assert!(v.contains("free on the 14th"));
        assert!(v.contains("still scoped to Q4"));
        // The raw fence must never reach the card.
        assert!(!v.contains("aa:assumes"), "raw fence leaked: {v}");
        // Row-1 verbs untouched — the annotation is display-only.
        assert!(v.contains("aa:act-a1:approve"));
        assert!(v.contains("aa:act-a1:revise"));
    }

    #[test]
    fn assumes_marker_round_trips() {
        let draft = "Hi,\n\nThe 14th works.\n\nBest,\nN";
        let facts = vec![
            "you're free on the 14th - not verified against calendar".to_string(),
            "the kickoff is still scoped to Q4".to_string(),
        ];
        let marked = append_assumes_marker(draft, &facts);
        let (human, got) = split_assumes(&marked);
        assert_eq!(human, draft);
        assert_eq!(got, facts);
    }

    #[test]
    fn assumes_marker_with_no_facts_is_byte_identical() {
        let draft = "Just a normal reply.";
        assert_eq!(append_assumes_marker(draft, &[]), draft);
        assert_eq!(append_assumes_marker(draft, &["  ".to_string()]), draft);
        let (human, facts) = split_assumes(draft);
        assert_eq!(human, draft);
        assert!(facts.is_empty());
    }

    #[test]
    fn assumes_marker_caps_at_five_lines() {
        let facts: Vec<String> = (0..9).map(|i| format!("fact {i}")).collect();
        let (_, got) = split_assumes(&append_assumes_marker("body", &facts));
        assert_eq!(got.len(), MAX_ASSUMES);
    }

    #[test]
    fn malformed_assumes_fence_is_treated_as_absent() {
        // Open fence, no close ⇒ raw text back, no field on the card.
        let broken = format!("body\n\n{ASSUMES_OPEN}you're free on the 14th");
        let (human, facts) = split_assumes(&broken);
        assert_eq!(human, broken);
        assert!(facts.is_empty());
        let v = json(&approval_message("act-a2", &email(), &broken, 0));
        assert!(!v.contains("Assumes\""), "half-parsed fence rendered a field: {v}");
    }

    #[test]
    fn assumes_fence_is_spliced_so_later_markers_survive() {
        // Production ordering is body → assumes → [cc:] → needs-input. Every
        // one of them must still reach the card after both splitters run.
        let with_assumes = append_assumes_marker(
            "Hi,\n\nWorks for me.",
            &["you're free on the 14th".to_string()],
        );
        let with_cc = format!("{with_assumes}\n\n[cc: will@example.com]");
        let body = append_needs_input_marker(
            &with_cc,
            &[("share_doc".to_string(), "the pitch deck".to_string())],
        );

        let (human, needs) = split_needs_input(&body);
        let (human, facts) = split_assumes(&human);
        assert_eq!(needs.len(), 1);
        assert_eq!(facts, vec!["you're free on the 14th".to_string()]);
        assert!(human.contains("Works for me."));
        assert!(human.contains("[cc: will@example.com]"), "cc marker spliced away: {human}");

        let v = json(&approval_message("act-a3", &email(), &body, 0));
        assert!(v.contains("Assumes"));
        assert!(v.contains("free on the 14th"));
        assert!(v.contains("Needs your input"));
        assert!(v.contains("[cc: will@example.com]"));
        assert!(!v.contains("aa:assumes"));
        assert!(!v.contains("aa:needs-input"));
    }

    #[test]
    fn strip_assumes_for_send_also_drops_an_unclosed_fence() {
        // The tolerant splitter leaves a malformed fence in place; a body
        // headed for a real inbox must be scrubbed anyway.
        let closed = append_assumes_marker("Hello.", &["a guess".to_string()]);
        assert_eq!(strip_assumes_for_send(&closed), "Hello.");
        let unclosed = format!("Hello.\n\n{ASSUMES_OPEN}a guess");
        assert_eq!(strip_assumes_for_send(&unclosed), "Hello.");
        assert_eq!(strip_assumes_for_send("Hello."), "Hello.");
    }

    #[test]
    fn fill_ask_modal_has_one_input_per_ask_keyed_by_kind() {
        let needs = vec![
            NeedsInput {
                kind: "scheduling".into(),
                text: "a 30-min call next week".into(),
            },
            NeedsInput {
                kind: "share_doc".into(),
                text: "the data room".into(),
            },
        ];
        let modal = fill_ask_modal("act-3", &needs);
        let v = serde_json::to_string(&modal).expect("modal serializes");
        assert!(v.contains("aa:act-3:fill_ask_modal"));
        assert!(v.contains("ask0:scheduling"));
        assert!(v.contains("ask1:share_doc"));
    }

    #[test]
    fn fill_feedback_lists_each_value_and_anti_placeholder_rule() {
        let filled = vec![
            (
                NeedsInput {
                    kind: "share_doc".into(),
                    text: "the Q4 deck".into(),
                },
                "https://drive.google.com/file/Q4".to_string(),
            ),
            (
                NeedsInput {
                    kind: "intro".into(),
                    text: "Sarah Chen".into(),
                },
                "Sure, happy to connect you.".to_string(),
            ),
        ];
        let fb = fill_feedback(&filled);
        assert!(fb.contains("Do NOT write a"));
        assert!(fb.contains("EXACTLY as given"));
        assert!(fb.contains("Document link"));
        assert!(fb.contains("the Q4 deck"));
        assert!(fb.contains("drive.google.com/file/Q4"));
        assert!(fb.contains("Introduction"));
        assert!(fb.contains("Sarah Chen"));
    }

    // ---------------------------------------------------------------------
    // #501: schedule row / scheduled notice / custom-time modal.
    // ---------------------------------------------------------------------

    /// Count action rows in a serialized CreateMessage.
    fn row_count(m: &CreateMessage) -> usize {
        let v: serde_json::Value =
            serde_json::from_str(&json(m)).expect("card json parses");
        v["components"].as_array().map(|a| a.len()).unwrap_or(0)
    }

    #[test]
    fn schedule_row_is_unconditional_even_at_cap() {
        // At the refine cap the quick-refine menu is dropped — but a gmail
        // card must STILL be schedulable (#501).
        let at_cap = json(&approval_message(
            "act-s1",
            &email(),
            "plain draft",
            MAX_REDRAFT_ITERATIONS,
        ));
        assert!(at_cap.contains("aa:act-s1:schedule_pick"));
        assert!(!at_cap.contains("quick_refine"), "cap must drop the menu");
        // And on a fresh card alongside the quick-refine menu.
        let fresh = json(&approval_message("act-s1", &email(), "plain draft", 0));
        assert!(fresh.contains("aa:act-s1:schedule_pick"));
        assert!(fresh.contains("quick_refine"));
    }

    /// #927 — Approve (which runs the merge) and Skip are the only verbs the
    /// merge handler implements, so they are the only ones rendered.
    #[test]
    fn identity_merge_card_offers_only_approve_and_skip() {
        let mut e = email();
        e.kind = "identity_merge".into();
        let v = json(&approval_message("act-m1", &e, "evidence", 0));
        assert!(v.contains("aa:act-m1:approve") && v.contains("Approve & Merge"));
        assert!(v.contains("aa:act-m1:skip") && !v.contains(":revise"), "nothing to redraft");
        assert_eq!(row_count(&approval_message("a", &email(), "d", 0)), 3, "other kinds unchanged");
    }

    #[test]
    fn schedule_row_absent_on_unschedulable_cards() {
        // The scheduled pipeline is a deferred Gmail send_draft — cards the
        // engine could never fire must not offer the control (#501 review).
        for platform in ["discord", "slack", "telegram", "github", "gcal"] {
            let mut e = email();
            e.platform = platform.into();
            let v = json(&approval_message("act-s6", &e, "d", 0));
            assert!(
                !v.contains("schedule_pick"),
                "{platform} card must not carry the schedule row"
            );
            // The rest of the card is untouched.
            assert!(v.contains("aa:act-s6:approve"));
        }
        // LinkedIn rides platform "gmail" on legacy rows — the entity prefix
        // is the tell.
        let mut e = email();
        e.account_entity_id = Some("linkedin:urn123".into());
        let v = json(&approval_message("act-s6", &e, "d", 0));
        assert!(!v.contains("schedule_pick"), "linkedin-entity card gated");
    }

    #[test]
    fn worst_case_card_is_four_rows_with_schedule_last() {
        // needs-input + under-cap: buttons + FillAsk + QuickRefine + Schedule
        // = 4 of Discord's 5 rows — the documented worst case.
        let marked = append_needs_input_marker(
            "draft",
            &[("share_doc".into(), "the deck".into())],
        );
        let msg = approval_message("act-s2", &email(), &marked, 0);
        assert_eq!(row_count(&msg), 4);
        let v = json(&msg);
        assert!(v.contains("aa:act-s2:schedule_pick"));
        assert!(v.contains("aa:act-s2:fill_ask"));
        assert!(v.contains("quick_refine"));
        // Plain fresh card: buttons + QuickRefine + Schedule = 3 rows.
        assert_eq!(row_count(&approval_message("a", &email(), "d", 0)), 3);
    }

    #[test]
    fn schedule_row_carries_all_symbolic_tokens() {
        let v = json(&approval_message("act-s3", &email(), "d", 0));
        for (_, value) in SCHEDULE_OPTIONS {
            assert!(v.contains(value), "missing schedule option value {value}");
        }
        assert!(v.contains("Schedule send…"));
    }

    #[test]
    fn scheduled_notice_has_exactly_three_buttons_and_no_approve() {
        let msg = scheduled_notice_message(
            "act-s4",
            &email(),
            "Mon Sep 1, 9:00 AM",
            1_756_716_000_000,
            &email().from,
        );
        let v: serde_json::Value =
            serde_json::from_str(&json(&msg)).expect("notice json parses");
        let rows = v["components"].as_array().expect("components");
        assert_eq!(rows.len(), 1, "notice has ONE button row");
        let buttons = rows[0]["components"].as_array().expect("buttons");
        assert_eq!(buttons.len(), 3, "exactly Send now instead | Back to queue | Cancel");
        let s = json(&msg);
        assert!(s.contains("aa:act-s4:send_now"));
        assert!(s.contains("aa:act-s4:back_to_queue"));
        assert!(s.contains("aa:act-s4:cancel_schedule"));
        // Must NOT carry the approval-card verbs — the draft is already
        // approved-for-later.
        assert!(!s.contains(":approve"));
        assert!(!s.contains(":revise"));
        assert!(!s.contains(":skip"));
        assert!(s.contains("Mon Sep 1, 9:00 AM"));
        assert!(s.contains("This email is armed and will send automatically."));
        assert!(s.contains("<t:1756716000:F>"));
        assert!(s.contains("<t:1756716000:R>"));
        assert!(s.contains("Send now instead"));
    }

    #[test]
    fn scheduled_notice_renders_the_resolved_to_not_the_card_from() {
        // #473 routing override (#501 review): the notice must show the
        // ACTUAL recipient the caller resolved from the envelope, not the
        // card's From line.
        let msg = scheduled_notice_message("act-s7", &email(), "t", 1_700_000_000_000, "omer@example.com");
        let v = json(&msg);
        assert!(v.contains("omer@example.com"));
        assert!(!v.contains("peer@example.com"), "card From must not leak into To");
    }

    #[test]
    fn schedule_modal_has_single_when_input() {
        let modal = schedule_modal("act-s5");
        let v = serde_json::to_string(&modal).expect("modal serializes");
        assert!(v.contains("aa:act-s5:schedule_modal"));
        assert!(v.contains("\"when\""));
        assert!(v.contains("tomorrow 9am"));
    }

    #[test]
    fn label_for_covers_all_kinds() {
        assert_eq!(label_for("scheduling"), "Proposed meeting time");
        assert_eq!(label_for("calendly"), "Booking link");
        assert_eq!(label_for("meeting_link"), "Video-call link");
        assert_eq!(label_for("share_doc"), "Document link");
        assert_eq!(label_for("intro"), "Introduction");
        assert_eq!(label_for("???"), "Detail");
    }
}
