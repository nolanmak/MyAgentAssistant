//! Fill-blanks-only person-page upsert for CRM ingestion (#61, #62, #64).
//!
//! Three ingestion sources — LinkedIn 1st-degree connections, Google /
//! CardDAV contacts, and email-signature parsing — all need to merge
//! self-reported third-party facts into `wiki/people/<slug>.md` under one
//! **iron rule**: *never overwrite, never invent.* An empty wiki field may be
//! filled; a non-empty one is left exactly as the human (or a higher-trust
//! source) left it. Running any ingest twice is a no-op.
//!
//! This module is the single deterministic, unit-tested implementation of
//! that merge. It owns *only* the in-memory string transform — no IO, no LLM,
//! no network. Callers (`augmentagent-channel-linkedin`, `-contacts`,
//! `-email`) decide *what* to merge and *whether* to write; this decides
//! *how* a merge changes the page text and *whether anything changed*.
//!
//! Page model:
//!
//! - **YAML frontmatter** carries the machine-readable `identities:` block
//!   ([`crate::identity::Identities`]). We append missing identity keys there
//!   so [`IdentityIndex`](crate::identity::IdentityIndex) keeps resolving.
//! - **Markdown body** carries human-facing facts under headed sections. We
//!   maintain a `## Profile` section (Role / Company / LinkedIn URL /
//!   Connected / Phone / Address / Website) and a `## Source` provenance
//!   section. A key already present with a non-empty value is never touched.
//!
//! New stub pages get a minimal frontmatter (`kind: person`) + the two
//! sections so the very next [`IdentityIndex::build`] resolves them.

use std::collections::BTreeMap;

/// One fill-blanks-only patch against a single person page.
///
/// Every field is "set only if the page has no value for it". `identities`
/// are merged into frontmatter; `profile` rows into the `## Profile` section;
/// `sources` are *appended* to `## Source` (provenance is additive — multiple
/// imports legitimately each leave a line, deduped on exact text).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PersonPatch {
    /// Display name — only used when creating a brand-new stub (`# <name>`).
    pub display_name: Option<String>,
    /// `platform -> id` identity pairs for the frontmatter `identities:` map.
    /// `email` is multi-valued; everything else is single-valued (matches
    /// [`crate::identity::Identities`]).
    pub identities: Vec<(String, String)>,
    /// Ordered profile rows rendered as `- **Key:** value` under `## Profile`.
    /// Order is preserved for *new* keys; existing keys keep their position.
    pub profile: Vec<(String, String)>,
    /// Provenance lines appended to `## Source` (deduped on exact text).
    pub sources: Vec<String>,
}

impl PersonPatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Add an identity pair. `platform` is one of the keys
    /// [`crate::identity::Identities`] understands (`email`, `linkedin`,
    /// `discord`, `twitter`, `slack`, `whatsapp`, `instagram`, plus the
    /// CRM-only `phone` / `address`). Empty ids are dropped.
    pub fn identity(mut self, platform: impl Into<String>, id: impl Into<String>) -> Self {
        let id = id.into();
        if !id.trim().is_empty() {
            self.identities.push((platform.into(), id));
        }
        self
    }

    /// Add a profile row. Empty values are dropped (we never write a blank
    /// `- **Role:**` line — that would defeat fill-blanks detection).
    pub fn profile_row(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let value = value.into();
        if !value.trim().is_empty() {
            self.profile.push((key.into(), value));
        }
        self
    }

    pub fn source(mut self, line: impl Into<String>) -> Self {
        let line = line.into();
        if !line.trim().is_empty() {
            self.sources.push(line);
        }
        self
    }

    /// True if the patch carries nothing to write.
    pub fn is_empty(&self) -> bool {
        self.identities.is_empty() && self.profile.is_empty() && self.sources.is_empty()
    }
}

/// What [`merge_person_page`] changed. The boolean `changed` gates whether the
/// caller writes the file at all (no-op merges must not churn git / mtime).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeResult {
    /// The full new page text. Equal to the input when `!changed`.
    pub content: String,
    /// True iff at least one field/identity/source line was actually added.
    pub changed: bool,
    /// True iff there was no prior page (a stub was created).
    pub created: bool,
    /// Human-readable list of what was filled, for the dry-run JSON dump and
    /// the Discord summary card ("Role, Company, linkedin").
    pub filled: Vec<String>,
}

/// Fill-blanks-only merge of `patch` into `existing` page text.
///
/// - `existing == None` → a fresh stub is created with frontmatter +
///   `## Profile` + `## Source`.
/// - A profile key already present with a non-empty value is **left
///   untouched** (even if `patch` carries a different value — we never
///   overwrite third-party self-reported data over the human's).
/// - An identity key already present in frontmatter is left untouched;
///   `email` is union-merged (multi-valued) but never deduped-away.
/// - `## Source` lines are appended unless the exact line already exists.
///
/// Idempotent: `merge(merge(p)) == merge(p)` and the second call reports
/// `changed == false`.
pub fn merge_person_page(existing: Option<&str>, patch: &PersonPatch) -> MergeResult {
    match existing {
        None => create_stub(patch),
        Some(src) => merge_into_existing(src, patch),
    }
}

fn create_stub(patch: &PersonPatch) -> MergeResult {
    let name = patch.display_name.as_deref().unwrap_or("Unknown");

    let mut fm = String::from("---\nkind: person\n");
    if !patch.identities.is_empty() {
        fm.push_str("identities:\n");
        for (plat, id) in collapse_identities(&patch.identities) {
            if is_multi_valued(&plat) {
                fm.push_str(&format!("  {plat}:\n"));
                for e in id.split('\u{1f}') {
                    fm.push_str(&format!("    - {}\n", yaml_scalar(e)));
                }
            } else {
                fm.push_str(&format!("  {plat}: {}\n", yaml_scalar(&id)));
            }
        }
    }
    fm.push_str("---\n");

    let mut body = format!("\n# {name}\n\n## Profile\n\n");
    let mut filled: Vec<String> = Vec::new();
    for (k, v) in dedupe_profile(&patch.profile) {
        body.push_str(&format!("- **{k}:** {v}\n"));
        filled.push(k);
    }
    body.push_str("\n## Source\n\n");
    for s in dedupe_preserve_order(&patch.sources) {
        body.push_str(&format!("- {s}\n"));
    }
    for (plat, _) in collapse_identities(&patch.identities) {
        filled.push(plat);
    }

    MergeResult {
        content: format!("{fm}{body}"),
        changed: true,
        created: true,
        filled,
    }
}

fn merge_into_existing(src: &str, patch: &PersonPatch) -> MergeResult {
    let mut filled: Vec<String> = Vec::new();

    // 1. Frontmatter identity merge (fill-blanks; email is union).
    let (mut page, fm_filled) = merge_frontmatter_identities(src, &patch.identities);
    filled.extend(fm_filled);

    // 2. `## Profile` rows — add only keys that are absent / blank.
    let (page2, prof_filled) = merge_section_rows(&page, "Profile", &patch.profile);
    page = page2;
    filled.extend(prof_filled);

    // 3. `## Source` provenance — append unseen lines.
    let (page3, src_added) = append_source_lines(&page, &patch.sources);
    page = page3;
    if src_added {
        filled.push("source".to_string());
    }

    let changed = page != src;
    MergeResult {
        content: page,
        changed,
        created: false,
        filled,
    }
}

/// Identity platforms that are YAML lists in `schema/wiki-skill.md`
/// (`email: [..]`, `phone: [..]`, `imessage: [..]`). Everything else is a
/// scalar. Writer and reader (`identity::Identities`) must agree on this
/// set — a scalar written under a list key makes the whole page unparseable
/// and silently drops it from the identity index.
const MULTI_VALUED: &[&str] = &["email", "phone", "imessage"];

fn is_multi_valued(platform: &str) -> bool {
    MULTI_VALUED.contains(&platform)
}

/// Collapse a `Vec<(platform,id)>` into ordered unique platforms. For the
/// multi-valued platforms we keep every distinct value joined by US
/// (`\u{1f}`) — an in-band separator that can't appear in an address or
/// phone — so the caller renders a YAML list. All other platforms keep the
/// first-seen value (single-valued).
fn collapse_identities(pairs: &[(String, String)]) -> Vec<(String, String)> {
    let mut order: Vec<String> = Vec::new();
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for (plat, id) in pairs {
        let id = id.trim();
        if id.is_empty() {
            continue;
        }
        if is_multi_valued(plat) {
            let entry = map.entry(plat.clone()).or_default();
            let already = entry.split('\u{1f}').any(|e| e.eq_ignore_ascii_case(id));
            if !already {
                if entry.is_empty() {
                    *entry = id.to_string();
                } else {
                    entry.push('\u{1f}');
                    entry.push_str(id);
                }
            }
        } else {
            map.entry(plat.clone()).or_insert_with(|| id.to_string());
        }
        if !order.contains(plat) {
            order.push(plat.clone());
        }
    }
    order
        .into_iter()
        .map(|p| {
            let v = map.get(&p).cloned().unwrap_or_default();
            (p, v)
        })
        .collect()
}

/// Merge identity pairs into the YAML frontmatter `identities:` block,
/// fill-blanks-only. Returns the new page + the list of platforms newly
/// added. Single-valued platforms already present are skipped; `email`
/// gains only addresses not already listed (case-insensitive).
fn merge_frontmatter_identities(
    src: &str,
    pairs: &[(String, String)],
) -> (String, Vec<String>) {
    if pairs.is_empty() {
        return (src.to_string(), Vec::new());
    }
    let collapsed = collapse_identities(pairs);

    let Some((fm_inner, close_off)) = split_frontmatter(src) else {
        // No frontmatter — synthesize a minimal one at the very top so the
        // identity index can resolve this page. Body is preserved verbatim.
        let mut fm = String::from("---\nkind: person\nidentities:\n");
        let mut added = Vec::new();
        for (plat, id) in &collapsed {
            if is_multi_valued(plat) {
                fm.push_str(&format!("  {plat}:\n"));
                for e in id.split('\u{1f}') {
                    fm.push_str(&format!("    - {}\n", yaml_scalar(e)));
                }
            } else {
                fm.push_str(&format!("  {plat}: {}\n", yaml_scalar(id)));
            }
            added.push(plat.clone());
        }
        fm.push_str("---\n");
        return (format!("{fm}{src}"), added);
    };

    let fm_inner = fm_inner.to_string();
    let mut added: Vec<String> = Vec::new();
    let mut new_lines: Vec<String> = Vec::new();

    let has_identities_key = fm_inner
        .lines()
        .any(|l| l.trim_end() == "identities:" || l.starts_with("identities:"));

    // List-valued keys already on the page are extended *in place* (a legacy
    // scalar, a flow list or a block list — whichever shape is there), so a
    // new value never lands after an unrelated key where it would be invalid
    // YAML. Absent keys and scalar keys are appended before the closing `---`.
    let mut fm_new = fm_inner.clone();
    for (plat, id) in &collapsed {
        if is_multi_valued(plat) {
            let existing = extract_list_values(&fm_new, plat);
            let fresh: Vec<String> = id
                .split('\u{1f}')
                .filter(|v| !existing.iter().any(|x| x.eq_ignore_ascii_case(v)))
                .map(str::to_string)
                .collect();
            if fresh.is_empty() {
                continue;
            }
            match extend_list_key(&fm_new, plat, &fresh) {
                Some(rewritten) => fm_new = rewritten,
                None => new_lines.push(format!("  {plat}: {}", flow_list(&fresh))),
            }
            added.push(plat.clone());
        } else if !identity_key_present(&fm_new, plat) {
            new_lines.push(format!("  {plat}: {}", yaml_scalar(id)));
            added.push(plat.clone());
        }
    }

    if fm_new == fm_inner && new_lines.is_empty() {
        return (src.to_string(), Vec::new());
    }

    let mut insert = String::new();
    if !has_identities_key && !new_lines.is_empty() {
        insert.push_str("identities:\n");
    }
    if !new_lines.is_empty() {
        insert.push_str(&new_lines.join("\n"));
        insert.push('\n');
    }

    let mut out = String::with_capacity(src.len() + insert.len() + 16);
    out.push_str("---\n");
    out.push_str(&fm_new);
    out.push_str(&insert);
    out.push_str(&src[close_off..]);
    (out, added)
}

/// `[a, b]` with each value YAML-quoted only when needed.
fn flow_list(values: &[String]) -> String {
    let items: Vec<String> = values.iter().map(|v| yaml_scalar(v)).collect();
    format!("[{}]", items.join(", "))
}

/// Add `fresh` values to the existing `key:` entry inside `fm_inner`,
/// touching only that key's lines. Handles every shape found on disk:
///
/// - block list (`key:` + `  - v` items) — items appended after the last one
/// - flow list (`key: [a, b]`) — rewritten as one flow list
/// - legacy scalar (`key: "+1…"`) — promoted to a flow list `[old, new]`
///
/// `None` when the key is absent; the caller appends a fresh list. The
/// trailing-newline state of `fm_inner` is preserved.
fn extend_list_key(fm_inner: &str, key: &str, fresh: &[String]) -> Option<String> {
    let lines: Vec<&str> = fm_inner.lines().collect();
    let head_idx = lines.iter().position(|l| {
        let t = l.trim_start();
        t == format!("{key}:") || t.starts_with(&format!("{key}: "))
    })?;
    let head = lines[head_idx];
    let indent = &head[..head.len() - head.trim_start().len()];
    let rest = head.trim_start()[key.len() + 1..].trim();

    let mut out: Vec<String> = lines[..head_idx].iter().map(|l| l.to_string()).collect();
    let mut resume = head_idx + 1;
    if rest.is_empty() {
        // Block list: keep the header and existing items, append after them.
        out.push(head.to_string());
        let mut item_indent = format!("{indent}  ");
        while resume < lines.len() {
            let l = lines[resume];
            let lead = &l[..l.len() - l.trim_start().len()];
            if lead.len() > indent.len() && l.trim_start().starts_with("- ") {
                item_indent = lead.to_string();
                out.push(l.to_string());
                resume += 1;
            } else {
                break;
            }
        }
        for v in fresh {
            out.push(format!("{item_indent}- {}", yaml_scalar(v)));
        }
    } else {
        let mut all: Vec<String> = if rest.starts_with('[') {
            split_flow_list(rest)
        } else {
            vec![unquote(rest).to_string()]
        };
        all.extend(fresh.iter().cloned());
        out.push(format!("{indent}{key}: {}", flow_list(&all)));
    }
    out.extend(lines[resume..].iter().map(|l| l.to_string()));

    let mut joined = out.join("\n");
    if fm_inner.ends_with('\n') {
        joined.push('\n');
    }
    Some(joined)
}

fn unquote(v: &str) -> &str {
    let v = v.trim();
    v.strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .or_else(|| v.strip_prefix('\'').and_then(|r| r.strip_suffix('\'')))
        .unwrap_or(v)
}

fn split_flow_list(rest: &str) -> Vec<String> {
    rest.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|part| unquote(part).to_string())
        .filter(|v| !v.is_empty())
        .collect()
}

/// Split into `(frontmatter_inner, byte_offset_of_closing_delim)`.
/// Mirrors `migrate::split_frontmatter` but kept local so `crm` has no
/// intra-crate coupling to the migration tool.
fn split_frontmatter(page: &str) -> Option<(&str, usize)> {
    let rest = page.strip_prefix("---\n")?;
    let mut offset = 4;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches('\n');
        if trimmed == "---" {
            let inner = &page[4..offset];
            return Some((inner, offset));
        }
        offset += line.len();
    }
    None
}

fn identity_key_present(fm_inner: &str, key: &str) -> bool {
    let prefix = format!("{key}:");
    fm_inner.lines().any(|l| {
        let t = l.trim_start();
        t == prefix || t.starts_with(&format!("{key}: "))
    })
}

/// Pull the existing values of a list-valued `key:` — block items
/// (`    - v`), an inline flow list (`key: [a, b]`) or a legacy bare scalar
/// (`key: "v"`, written before the key became a list).
fn extract_list_values(fm_inner: &str, key: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_block = false;
    let prefix = format!("{key}:");
    for line in fm_inner.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(&prefix) {
            if !rest.is_empty() && !rest.starts_with(' ') {
                // `keyword:` is a prefix of a longer key (`phone_home:`)
                in_block = false;
                continue;
            }
            let rest = rest.trim();
            if rest.starts_with('[') {
                out.extend(split_flow_list(rest));
                in_block = false;
            } else if rest.is_empty() {
                in_block = true;
            } else {
                out.push(unquote(rest).to_string());
                in_block = false;
            }
            continue;
        }
        if in_block {
            if let Some(item) = trimmed.strip_prefix("- ") {
                out.push(unquote(item).to_string());
            } else if !trimmed.is_empty() {
                in_block = false;
            }
        }
    }
    out
}

/// Result of [`merge_stub_into`]: the rewritten target page plus what moved.
#[derive(Debug)]
pub struct StubMergeResult {
    pub content: String,
    pub changed: bool,
    /// Platforms / sections the merge actually filled on the target.
    pub moved: Vec<String>,
}

/// Fold an auto-created contact stub (`<name>_at_contact.md`, produced by the
/// contacts / iMessage backfills) into a canonical person page, as one pure
/// string transform. Moves the stub's list-valued identities (the only kind a
/// stub ever carries — see `MULTI_VALUED`) and its `## Source` provenance
/// lines into the target via the same fill-blanks `merge_person_page` path,
/// and appends a `Merged from …` provenance line. Deleting the stub file and
/// repointing `identity_phone` are the caller's IO (`person merge` in the
/// CLI) — this module never touches disk.
///
/// Idempotent: applying twice reports `changed == false` the second time.
pub fn merge_stub_into(target_src: &str, stub_src: &str, stub_slug: &str) -> StubMergeResult {
    let mut patch = PersonPatch::new();
    if let Some((fm_inner, _)) = split_frontmatter(stub_src) {
        for plat in MULTI_VALUED {
            for v in extract_list_values(fm_inner, plat) {
                patch = patch.identity(*plat, v);
            }
        }
    }
    for line in source_lines(stub_src) {
        patch = patch.source(line);
    }
    patch = patch.source(format!("Merged from {stub_slug}.md (owner-approved)"));
    let r = merge_person_page(Some(target_src), &patch);
    StubMergeResult {
        content: r.content,
        changed: r.changed,
        moved: r.filled,
    }
}

/// The `- …` items of a page's `## Source` section, without the bullet. Public
/// so a merge proposal can SHOW this provenance on its approval card (#927).
pub fn source_lines(page: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_source = false;
    for line in page.lines() {
        let t = line.trim();
        if t == "## Source" {
            in_source = true;
            continue;
        }
        if in_source {
            if t.starts_with("## ") {
                break;
            }
            if let Some(item) = t.strip_prefix("- ") {
                out.push(item.to_string());
            }
        }
    }
    out
}

/// YAML-quote a scalar only when it could be misparsed (`:` `#` leading `-`,
/// etc). Emails / urns / phone are safe bare in practice but we quote
/// defensively when in doubt.
fn yaml_scalar(s: &str) -> String {
    let needs = s.is_empty()
        || s.contains(": ")
        || s.contains(" #")
        // Leading `+` / `-` (phone numbers) — quoted so no YAML parser ever
        // coerces `+14155550100` toward a number-ish scalar.
        || s.starts_with(['-', '+', '?', '*', '&', '!', '@', '`', '"', '\'', '[', '{', '|', '>'])
        || s.contains('\n');
    if needs {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

/// Ensure a `## <heading>` section exists and contains a `- **Key:** value`
/// row for every key in `rows` that is *not already present with a non-empty
/// value*. Returns the new page + the list of keys actually added.
fn merge_section_rows(
    page: &str,
    heading: &str,
    rows: &[(String, String)],
) -> (String, Vec<String>) {
    if rows.is_empty() {
        return (page.to_string(), Vec::new());
    }
    let rows = dedupe_profile(rows);

    let header = format!("## {heading}");
    let mut added: Vec<String> = Vec::new();

    // Locate the section span: from the heading line to the next `## ` (or EOF).
    let lines: Vec<&str> = page.lines().collect();
    let sec_start = lines.iter().position(|l| l.trim_end() == header);

    let existing_keys: Vec<String> = if let Some(start) = sec_start {
        let mut keys = Vec::new();
        for l in &lines[start + 1..] {
            if l.starts_with("## ") {
                break;
            }
            if let Some(k) = parse_row_key(l) {
                keys.push(k.to_ascii_lowercase());
            }
        }
        keys
    } else {
        Vec::new()
    };

    let mut to_add: Vec<String> = Vec::new();
    for (k, v) in &rows {
        if !existing_keys.contains(&k.to_ascii_lowercase()) {
            to_add.push(format!("- **{k}:** {v}"));
            added.push(k.clone());
        }
    }
    if to_add.is_empty() {
        return (page.to_string(), Vec::new());
    }

    let new_page = match sec_start {
        Some(start) => {
            // Insert after the last existing row in the section (before the
            // blank line that precedes the next heading, or section end).
            let mut insert_at = start + 1;
            for (i, l) in lines.iter().enumerate().skip(start + 1) {
                if l.starts_with("## ") {
                    break;
                }
                if parse_row_key(l).is_some() {
                    insert_at = i + 1;
                }
            }
            let mut out: Vec<String> = lines[..insert_at].iter().map(|s| s.to_string()).collect();
            out.extend(to_add.iter().cloned());
            out.extend(lines[insert_at..].iter().map(|s| s.to_string()));
            join_preserving_trailing_nl(page, &out)
        }
        None => {
            // Append a fresh section at the end.
            let mut base = page.trim_end().to_string();
            base.push_str(&format!("\n\n{header}\n\n"));
            base.push_str(&to_add.join("\n"));
            base.push('\n');
            base
        }
    };
    (new_page, added)
}

/// Append every line in `sources` to a `## Source` section unless that exact
/// rendered bullet already exists anywhere in the page.
fn append_source_lines(page: &str, sources: &[String]) -> (String, bool) {
    if sources.is_empty() {
        return (page.to_string(), false);
    }
    let wanted = dedupe_preserve_order(sources);
    let to_add: Vec<String> = wanted
        .into_iter()
        .filter(|s| {
            let bullet = format!("- {s}");
            !page.lines().any(|l| l.trim_end() == bullet)
        })
        .collect();
    if to_add.is_empty() {
        return (page.to_string(), false);
    }

    let header = "## Source";
    let lines: Vec<&str> = page.lines().collect();
    let sec_start = lines.iter().position(|l| l.trim_end() == header);

    let new_page = match sec_start {
        Some(start) => {
            let mut insert_at = start + 1;
            for (i, l) in lines.iter().enumerate().skip(start + 1) {
                if l.starts_with("## ") {
                    break;
                }
                if l.trim_start().starts_with("- ") {
                    insert_at = i + 1;
                }
            }
            let mut out: Vec<String> =
                lines[..insert_at].iter().map(|s| s.to_string()).collect();
            for s in &to_add {
                out.push(format!("- {s}"));
            }
            out.extend(lines[insert_at..].iter().map(|s| s.to_string()));
            join_preserving_trailing_nl(page, &out)
        }
        None => {
            let mut base = page.trim_end().to_string();
            base.push_str("\n\n## Source\n\n");
            for s in &to_add {
                base.push_str(&format!("- {s}\n"));
            }
            base
        }
    };
    (new_page, true)
}

fn parse_row_key(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let rest = t.strip_prefix("- **")?;
    let end = rest.find(":**")?;
    Some(rest[..end].trim())
}

fn dedupe_profile(rows: &[(String, String)]) -> Vec<(String, String)> {
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for (k, v) in rows {
        let lk = k.to_ascii_lowercase();
        if v.trim().is_empty() || seen.contains(&lk) {
            continue;
        }
        seen.push(lk);
        out.push((k.clone(), v.trim().to_string()));
    }
    out
}

fn dedupe_preserve_order(items: &[String]) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for s in items {
        let s = s.trim().to_string();
        if s.is_empty() || seen.contains(&s) {
            continue;
        }
        seen.push(s.clone());
        out.push(s);
    }
    out
}

/// Re-join `lines` while restoring the original page's trailing-newline state
/// (`str::lines()` drops it).
fn join_preserving_trailing_nl(original: &str, lines: &[String]) -> String {
    let mut s = lines.join("\n");
    if original.ends_with('\n') {
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::IdentityIndex;
    use crate::layout::WikiLayout;

    #[test]
    fn creates_stub_with_sections_and_identities() {
        let patch = PersonPatch::new()
            .with_display_name("Jane Doe")
            .identity("linkedin", "jane-doe-123")
            .profile_row("Role", "Staff Engineer")
            .profile_row("Company", "Acme")
            .source("Imported from LinkedIn 1st-degree connections on 2026-05-18");
        let r = merge_person_page(None, &patch);
        assert!(r.created && r.changed);
        assert!(r.content.starts_with("---\nkind: person\n"));
        assert!(r.content.contains("linkedin: jane-doe-123"));
        assert!(r.content.contains("# Jane Doe"));
        assert!(r.content.contains("## Profile"));
        assert!(r.content.contains("- **Role:** Staff Engineer"));
        assert!(r.content.contains("- **Company:** Acme"));
        assert!(r.content.contains("## Source"));
        assert!(r
            .content
            .contains("- Imported from LinkedIn 1st-degree connections on 2026-05-18"));
    }

    #[test]
    fn idempotent_second_merge_is_noop() {
        let patch = PersonPatch::new()
            .with_display_name("Jane")
            .identity("linkedin", "jane-1")
            .profile_row("Role", "Eng")
            .source("Imported from LinkedIn on 2026-05-18");
        let first = merge_person_page(None, &patch);
        let second = merge_person_page(Some(&first.content), &patch);
        assert!(!second.changed, "second merge must be a no-op");
        assert_eq!(second.content, first.content);
        assert!(second.filled.is_empty());
    }

    #[test]
    fn never_overwrites_existing_profile_value() {
        let existing = "---\nkind: person\n---\n\n# Jane\n\n## Profile\n\n- **Role:** Founder\n\n## Source\n\n";
        let patch = PersonPatch::new().profile_row("Role", "Intern");
        let r = merge_person_page(Some(existing), &patch);
        assert!(!r.changed);
        assert!(r.content.contains("- **Role:** Founder"));
        assert!(!r.content.contains("Intern"));
    }

    #[test]
    fn fills_only_missing_profile_row() {
        let existing = "---\nkind: person\n---\n\n# Jane\n\n## Profile\n\n- **Role:** Founder\n";
        let patch = PersonPatch::new()
            .profile_row("Role", "Intern")
            .profile_row("Company", "Acme");
        let r = merge_person_page(Some(existing), &patch);
        assert!(r.changed);
        assert!(r.content.contains("- **Role:** Founder"));
        assert!(r.content.contains("- **Company:** Acme"));
        assert_eq!(r.filled, vec!["Company".to_string()]);
    }

    #[test]
    fn frontmatter_identity_fill_blanks_only() {
        let existing = "---\nkind: person\nidentities:\n  linkedin: old-handle\n---\n\n# Jane\n";
        let patch = PersonPatch::new()
            .identity("linkedin", "new-handle")
            .identity("phone", "+14155550100");
        let r = merge_person_page(Some(existing), &patch);
        assert!(r.changed);
        // linkedin already present → untouched.
        assert!(r.content.contains("linkedin: old-handle"));
        assert!(!r.content.contains("new-handle"));
        // phone was blank → added, as a list (schema/wiki-skill.md).
        assert!(r.content.contains("  phone: [\"+14155550100\"]\n"), "{}", r.content);
    }

    #[test]
    fn email_identity_is_union_merged() {
        let existing = "---\nkind: person\nidentities:\n  email:\n    - a@x.com\n---\n\n# Jane\n";
        let patch = PersonPatch::new()
            .identity("email", "a@x.com")
            .identity("email", "b@y.com");
        let r = merge_person_page(Some(existing), &patch);
        assert!(r.changed);
        assert!(r.content.contains("- a@x.com"));
        assert!(r.content.contains("- b@y.com"));
        // a@x.com not duplicated.
        assert_eq!(r.content.matches("a@x.com").count(), 1);
    }

    #[test]
    fn synthesizes_frontmatter_when_absent() {
        let existing = "# Jane\n\nSome notes.\n";
        let patch = PersonPatch::new().identity("linkedin", "jane-9");
        let r = merge_person_page(Some(existing), &patch);
        assert!(r.changed);
        assert!(r.content.starts_with("---\nkind: person\nidentities:\n"));
        assert!(r.content.contains("linkedin: jane-9"));
        assert!(r.content.contains("Some notes."));
    }

    #[test]
    fn appends_source_lines_dedup() {
        let existing = "---\nkind: person\n---\n\n# Jane\n\n## Source\n\n- Imported from LinkedIn on 2026-01-01\n";
        let patch = PersonPatch::new()
            .source("Imported from LinkedIn on 2026-01-01")
            .source("Imported from Google Contacts on 2026-05-18");
        let r = merge_person_page(Some(existing), &patch);
        assert!(r.changed);
        assert_eq!(
            r.content
                .matches("Imported from LinkedIn on 2026-01-01")
                .count(),
            1
        );
        assert!(r
            .content
            .contains("- Imported from Google Contacts on 2026-05-18"));
    }

    #[test]
    fn creates_profile_section_when_missing() {
        let existing = "---\nkind: person\n---\n\n# Jane\n\nFreeform note.\n";
        let patch = PersonPatch::new().profile_row("Phone", "+14155550100");
        let r = merge_person_page(Some(existing), &patch);
        assert!(r.changed);
        assert!(r.content.contains("## Profile"));
        assert!(r.content.contains("- **Phone:** +14155550100"));
        assert!(r.content.contains("Freeform note."));
    }

    #[test]
    fn empty_patch_never_changes() {
        let existing = "---\nkind: person\n---\n\n# Jane\n";
        let r = merge_person_page(Some(existing), &PersonPatch::new());
        assert!(!r.changed);
        assert_eq!(r.content, existing);
    }

    #[test]
    fn phone_and_imessage_stubs_are_lists_the_identity_index_can_read() {
        // Regression: the writer emitted `phone: "+1…"` while the reader
        // declares `Vec<String>`, so every page the iMessage backfill wrote
        // was skipped by the identity index with "expected a sequence".
        let patch = PersonPatch::new()
            .with_display_name("Nav")
            .identity("imessage", "+12012757410")
            .identity("phone", "+12012757410");
        let out = merge_person_page(None, &patch);
        assert!(
            out.content.contains("  imessage:\n    - \"+12012757410\"\n"),
            "{}",
            out.content
        );
        assert!(
            out.content.contains("  phone:\n    - \"+12012757410\"\n"),
            "{}",
            out.content
        );
        let dir = tempfile::TempDir::new().unwrap();
        let layout = WikiLayout::new(dir.path().to_path_buf());
        std::fs::create_dir_all(layout.people_dir()).unwrap();
        std::fs::write(layout.people_dir().join("nav.md"), &out.content).unwrap();
        let index = IdentityIndex::build(&layout).unwrap();
        assert_eq!(index.len(), 1, "page must parse");
        assert!(index.lookup("imessage", "+12012757410").is_some());
        assert!(index.lookup("phone", "+12012757410").is_some());
    }

    #[test]
    fn list_keys_are_extended_in_place_whatever_their_shape() {
        // flow list not last in the block, legacy scalar phone, block email
        let page = "---\nkind: person\nidentities:\n  email: [a@example.com]\n  phone: \"+15550001\"\n  linkedin: urn:li:1\n---\n\n# P\n";
        let patch = PersonPatch::new()
            .identity("email", "b@example.com")
            .identity("phone", "+15550002")
            .identity("imessage", "+15550002");
        let out = merge_person_page(Some(page), &patch);
        assert!(out.changed);
        assert!(out.content.contains("  email: [a@example.com, b@example.com]\n"), "{}", out.content);
        assert!(
            out.content.contains("  phone: [\"+15550001\", \"+15550002\"]\n"),
            "{}",
            out.content
        );
        assert!(out.content.contains("  linkedin: urn:li:1\n"), "{}", out.content);
        assert!(
            out.content.contains("  imessage: [\"+15550002\"]\n---\n"),
            "{}",
            out.content
        );
        // still valid YAML for the strict reader, and idempotent
        let dir = tempfile::TempDir::new().unwrap();
        let layout = WikiLayout::new(dir.path().to_path_buf());
        std::fs::create_dir_all(layout.people_dir()).unwrap();
        std::fs::write(layout.people_dir().join("p.md"), &out.content).unwrap();
        let index = IdentityIndex::build(&layout).unwrap();
        let hit = index.lookup("phone", "+15550001").expect("old phone kept");
        assert_eq!(hit.identities.phone, vec!["+15550001", "+15550002"]);
        assert_eq!(hit.identities.email, vec!["a@example.com", "b@example.com"]);
        let again = merge_person_page(Some(&out.content), &patch);
        assert!(!again.changed, "second merge must be a no-op:\n{}", again.content);
    }

    #[test]
    fn block_list_gets_items_appended_after_the_last_existing_item() {
        let page = "---\nkind: person\nidentities:\n  email:\n    - a@example.com\n  linkedin: urn:li:1\n---\n";
        let out = merge_person_page(Some(page), &PersonPatch::new().identity("email", "b@example.com"));
        assert_eq!(
            out.content,
            "---\nkind: person\nidentities:\n  email:\n    - a@example.com\n    - b@example.com\n  linkedin: urn:li:1\n---\n"
        );
    }

    #[test]
    fn stub_merge_moves_identities_and_sources_and_is_idempotent() {
        let target = "---\nkind: person\nkey: centra\nupdated: 2026-06-01\nidentities:\n  email: [org@example.com]\n---\n\n# Centra\n\n## Profile\n\n## Source\n\n- email thread\n";
        let stub = "---\nkind: person\nidentities:\n  imessage: [\"+12067598450\"]\n  phone: [\"+12067598450\"]\nupdated: 2026-08-25\n---\n\n# Landlord\n\n## Profile\n\n## Source\n\n- iMessage history: 23 messages through 2026-08-25 (SMS)\n";
        let r = merge_stub_into(target, stub, "landlord_at_contact");
        assert!(r.changed);
        assert!(r.content.contains("  imessage: [\"+12067598450\"]"), "{}", r.content);
        assert!(r.content.contains("  phone: [\"+12067598450\"]"), "{}", r.content);
        assert!(r.content.contains("email: [org@example.com]"), "target identity kept");
        assert!(r.content.contains("- iMessage history: 23 messages through 2026-08-25 (SMS)"));
        assert!(r.content.contains("- Merged from landlord_at_contact.md (owner-approved)"));
        assert!(r.content.contains("- email thread"), "target sources kept");
        // second application is a no-op
        let again = merge_stub_into(&r.content, stub, "landlord_at_contact");
        assert!(!again.changed, "must be idempotent:\n{}", again.content);
    }

    #[test]
    fn stub_merge_unions_into_existing_phone_list() {
        let target = "---\nkind: person\nidentities:\n  phone: [\"+15550001\"]\n---\n\n# P\n\n## Source\n";
        let stub = "---\nkind: person\nidentities:\n  phone: [\"+15550002\"]\n---\n\n# Q\n\n## Source\n";
        let r = merge_stub_into(target, stub, "q_at_contact");
        assert!(r.content.contains("  phone: [\"+15550001\", \"+15550002\"]"), "{}", r.content);
    }

    #[test]
    fn identity_index_resolves_created_stub() {
        let dir = tempfile::TempDir::new().unwrap();
        let layout = WikiLayout::new(dir.path().to_path_buf());
        std::fs::create_dir_all(layout.people_dir()).unwrap();
        let patch = PersonPatch::new()
            .with_display_name("Jane Doe")
            .identity("linkedin", "jane-doe-xyz");
        let r = merge_person_page(None, &patch);
        std::fs::write(layout.people_dir().join("jane.md"), &r.content).unwrap();
        let idx = IdentityIndex::build(&layout).unwrap();
        assert!(idx.lookup("linkedin", "jane-doe-xyz").is_some());
    }
}
