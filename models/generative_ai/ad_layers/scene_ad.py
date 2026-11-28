"""포스터의 구도를 먼저 정하고 그림과 글자를 별도 레이어로 만듭니다."""

import argparse
import json
import re
from math import ceil, floor, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .art_direction import ANGLES, LIGHTING, SHOT_SCALES, THEMES, plan_art_direction
from .creative_brief import LAYOUT_ROLES, MAX_HEADLINE_LENGTH, TEXT_ROLES, ask_stage, plan_copy, required_text, validate_boxes
from .layers import compose, cutout_product, load_layout, load_settings, paste_region, save_layout, to_float
from .staging import on_gray, padded_box, source_parts, stage_group, stage_subject
from .text_render import minimum_font_size, render_texts
from .type_contrast import readable_text_colors
from .viewpoint import available_viewpoints, camera_view, place_camera_view


DESIGN = """Art-direct one complete Korean campaign poster before any artwork is generated. The first image
shows the actual product. Remaining images, if present, are references for hierarchy, rhythm, and the relationship
between photo, type, and color. Borrow design principles only. Never copy their text, brands, or artwork.
Use the real product details and campaign message to make one visible creative choice that ties together the
photography, graphic color and typography. Decide what wins the first glance. Give supporting copy distinct
scale and grouping. Use deliberately shaped negative space and allow type beside or over quiet photo areas when
it supports the concept. Avoid the catalog pattern of a centered rectangular image with a heading above and a
name/price footer. Use one visual decision that still reads at thumbnail size, such as a close-up of a real selling
detail or a strong axis linking photo and copy. Start with a shared photographic field where light or shape can
lead from the product into the copy. Use a separate graphic panel only when its edge or color plays an intentional
part in that direction. Place the two supporting text elements close enough to read as a group. Use a solid color accent only
if it continues a line or a color from the photograph, or gives supporting type an intentional shared field.
An accent can cross from the edge or sit in quiet space over the photo. Give the main line room to lead."""

DESIGN_SCHEMA = """A design object has "concept" (the visual idea and why it serves the message), "focus" (product, headline,
product_name, or price), "background" (#RRGGBB), "accent" (null, or an object with "shape" rectangle or ellipse,
"box", "color" #RRGGBB, and "layer" background, overlay or behind_subject), "artwork_box", "subject_box", "edge" (blend or framed),
"crop" (preserve or intentional), and "art_direction" (English
photographic framing, lighting, materials and background for the artwork region; no text or graphic overlays).
Use a background accent outside the artwork, or an overlay accent over quiet space without covering the product.
Use behind_subject when a shape should tuck behind the real silhouette. Give it a clear visible role linking the
feature and headline. The compositor hides only the part behind the subject and keeps the product visible.
Use the blended edge for a calm photo perimeter that can merge into a plain graphic field. A framed edge can
align a deliberate, distinct image panel with the type or cover the whole canvas. When the background above
or below the photo should read as the same continuous surface, prefer blend.
Use preserve for a complete product by default. Choose intentional only when a motivated crop emphasizes a real
detail and the shopper can still identify the product and any important label.
For an intentional detail shot include source_crop as [left, top, width, height] in whole percentages of the
tight product bounds in the original image. Select the specific detail the headline is about. Space around the
product in the original photo is outside these bounds. Use null or omit source_crop for a complete product.
Place intentional cut edges against a photo edge rather than leaving a sliced product floating inside open space.
If there is a quiet area for type over the artwork, say where it is and describe its light and texture in art_direction.
Include "boxes" and "styles", both keyed by headline, product_name and price.
All boxes are [left, top, width, height]. Use whole percentages of the poster from its top-left, except subject_box,
which is relative to the photograph itself. Put the product inside subject_box with the intended scale and space
around the edges. In the shared photo area, reserve space for the headline and the other copy outside the product.
Keep text boxes
inside the canvas, off one another, and large enough for the supplied copy. Text may overlap the artwork box
only where you plan quiet background. Group related text by alignment and use sufficient contrast.
For each style return "font" (one of the supplied fonts), "align" (left, center, right), "color" (#RRGGBB),
"size" (intended font size as a percentage of canvas width), and "max_lines" (1 or 2 for the
headline or product_name, 1 for price). If two lines are intentional, set "break_after" to the number of
space-separated words on the first line. Keep Korean phrases, quantities and product names intact at the
break. For the headline only, optionally set "depth" to "behind_subject" when the contour of the foreground
can deliberately cross a small outer part of a large headline. Every word must stay readable. Leave out
depth when the type belongs in clear space. For the headline only, optional emphasis is an object with phrase
(an exact contiguous phrase from the headline) and color (#RRGGBB). Use one accent when a visible color from
the photograph draws attention to the corresponding words. Keep sufficient contrast. Use null to omit it.
For a deliberately two-line headline, optional styles.headline.lines contains two objects in reading order.
Each has a non-overlapping box inside the headline's outer box, in whole poster percentages, and may override
size, align, color, font or depth. For an override, depth can be only front or behind_subject. Set break_after
once on the parent headline to choose the exact words on each line. Set distinct positions
or scales only when each clause has a visible role. Enlarge the words that carry the campaign message.
Use a quieter scale for generic lead-ins or reactions. Equal treatment works when both lines share the message.
If a prior design failed, address the reported issue."""

BLOCKING = """Review the first image as a full poster made from the real product silhouette and the exact
rendered typography. The flat background is a placeholder; judge placement, scale, spacing and how the
reader's eye moves. The second image is the source photograph. Any further images are composition references.
Name the specific element that physically connects the headline and subject. If reading the separate blocks
depends on a lighting effect that has not been generated, correct the underlying geometry now. Consider a
common center axis, a relationship to a particular part of the product, or a deliberate edge relationship
with a legible large headline. Use the treatment justified by this product and message.
When the structure is weak, move or resize the actual subject_box and the copy together. Avoid only changing
font size or color. Keep the full recognizable product, exact copy, and one clear group for supporting details.
Read the thumbnail from the first focal point to the offer. Check whether any solid accent makes supporting
information the first focal point. Let that happen when the offer is the approved message. Otherwise soften
or remove the accent and use alignment, spacing or weight to keep the offer together. Use rendered_text_bounds
to check the actual ink edges before the photograph is generated.
If present, follow the requested visual correction. Return JSON with reason (Korean) and revision (null when
the structure works, or an object containing the design fields that need to change)."""

BLOCKING_COMPARE = """Compare the two rendered drafts in the first two images. They use the same product
and copy and neither has final lighting. Any later images are composition references. Judge the visible
relationship between the headline and product, the way their sizes balance, and the placement of the
smaller information. Look at the actual space and ink, even if the proposal described a different effect.
Compare which element gets the first glance and whether any graphic distracts from the advertised detail.
Keep the original when the revision would merely trade one generic arrangement for another.
Return JSON with choice (original or revision) and a concrete Korean reason."""

OFFER_SCENE_COMPARE = """Compare two placement proofs before photography. The outlined edge in the first
proof marks where the photograph would end and the flat design background would begin. The second proof
lets one photograph continue beneath the selling information, with the same product scale and type positions.
The outline and flat colors are only planning guides. Use the pictured product and campaign message to decide
which arrangement is stronger. Choose continuous if a quiet part of the same real surface, light or setting
can carry the offer and connect it to the product. Keep original when separate visual sections support the
message better or the continued photograph would have no plausible space for legible type.
Return choice (original or continuous) and reason (Korean, naming the purpose of the boundary or the specific
surface and lighting that can continue beneath the information)."""

SOURCE_FRAMING = """Propose one close framing to compare with the whole available product before layout.
The first image is the isolated product in its tight bounds. The second is the original photo; any later images
are references. Another reviewer will choose between the two views after seeing the rendered crop. If the message
concerns one detail or a few neighboring features, show those features together much larger. Leave out parts that
do not help explain the message. A whole shot can be good and still deserve comparison with a close shot.
Keep a visible contour, feature or relationship that explains what the advertised product is even with its
name covered. If the source shows repeated items, center the crop on one identifiable instance unless their
arrangement is the advertised feature. Return null if a meaningful crop would break necessary labels,
shape, count, or the arrangement across the whole product, or if the request specifies keeping the full view.
The original photo may already cut an edge of the object; preserve the meaning of its available view.
Return JSON with source_crop (null or [left, top, width, height] in integer percentages of the first image)
and reason (in Korean, what the crop makes easier to notice, or why a crop cannot help)."""

SOURCE_DETAIL_OBSERVATION = """Describe only this cropped photograph. Gray padding is outside the crop.
Say how many instances of the apparent subject you see, which visible forms you can trace individually,
and which are cut off or overlap. Mark competing_fragments true if it mainly shows several similarly
prominent, incomplete instances so there is no visually dominant instance to read as the main subject.
A detail of one visually dominant instance or a clearly readable arrangement is not competing fragments.
Return JSON with observation (a concrete Korean sentence) and competing_fragments (boolean)."""

SOURCE_FRAMING_CHECK = """The first two images show the whole available product and the actual crop in equal-size frames.
Judge the second image on its own first. Name the visible shape, detail or relationship that tells a viewer
what is being sold without relying on the product name or the whole view. Return detail_preserves_identity
(boolean). Generic texture or overlapping cropped fragments without a clearly identifiable subject are
insufficient. An intentional macro is valid if a distinctive feature still identifies the product.
Then compare it with the first and third images and product_facts for important labels and assortment.
A per-item price or unit of sale does not imply that only one example can appear in a photograph.
Choose detail only if the subject remains identifiable and the enlargement materially helps the headline.
Return JSON with choice (whole or detail), detail_preserves_identity and a Korean reason naming what
identifies the subject in the crop or what the crop loses."""

HEADLINE_PRIORITY = """Given the approved message, the photograph without type and the exact headline lines,
identify which words should receive typographic emphasis. Let the viewer connect those words to the subject
or idea the picture actually puts first. A side prop may establish the occasion without being what the ad
sells. Keep the focus on the part that names or evokes the advertised item or benefit in that case.
A generic reaction or transition usually supports the idea. A specific occasion or benefit can lead if it is
the point of the campaign and the image supports it. The reading order does not determine emphasis.
Return JSON with phrase (exact contiguous space-separated words from one line, with Korean particles intact,
or null if both lines carry the message together) and reason (one Korean sentence)."""

FINAL_REVIEW = """Review the first poster, using the second image (its generated photo without type) and the reference
posters that follow. Keep the generated photo, position, crop and canvas color. Use the actual product bounds to
place type outside the product. Group the related text, give the main line a distinct scale, and judge contrast
using the actual photograph. Find the visible axis, gesture or light that connects photo and words. If there is
an unused band between them or the offer reads as unrelated fragments, move and scale the type as one composition.
Use rendered_text_bounds to judge the actual ink, not the empty text boxes. For stacked supporting copy,
leave less space between the name and price than between their group and the headline. Keep that group near
the product rather than letting all the type drift into a distant corner. For a one-line information group,
fit the first text box to its visible lettering before placing the next box. A shared baseline with excess
space between the two can still make them look unrelated.
Keep useful relationships that already exist. When the original title leads into the photo from above and
supporting copy closes the reading path below it, preserve that order unless the new placement has a stronger
visible reason. Examine the whole canvas before moving every text group into the same margin.
When the approved image and headline still feel like separate columns, consider whether one larger headline
can share the subject's axis or tuck behind a small part of its outline. Use depth only when it adds that relation.
Match the lettering to the visual mood and keep any confirmed line breaks. Use the supplied wording exactly.
When the photograph clearly supports one specific phrase, consider highlighting it with a color sampled visually
from the relevant feature. Put {phrase, color} only at revision.styles.headline.emphasis. The revision.accent
field is only for a graphic shape using shape, layer, box and color, or null to remove the graphic.
For a deliberate shape that links the real subject contour to the headline, use layer behind_subject. The
subject stays entirely visible. Keep both a visible part of the graphic and a part tucked behind the subject.
Copy whole space-separated words exactly, including attached Korean particles or punctuation. Omit the emphasis
if there is no suitable phrase. Use a single treatment and keep the accent legible against the actual background.
If the same bright color on supporting information competes with the headline, reduce its emphasis.
When a single headline box keeps both clauses far from the advertised feature, use an intentional break_after
and two styles.headline.lines objects. Select an integer break_after from headline_breaks, which lists the first
and second line text for each valid break. Each line has its own box in whole poster percentages inside the outer
headline box, and can override size, font, align, color or depth. Place the line that names the feature near it.
Use the other line to set up the message. Keep the two line boxes in reading order, vertical and non-overlapping.
Keep both lines readable at feed size. Leave lines null to use one shared headline style.
If a solid graphic panel is the strongest element by color contrast and the offer is secondary, soften or
remove the panel. Use a common alignment and gap for the supporting text instead of relying on a shared box.
If a hard photo boundary behind the words or between the heading and product cuts a shared background for no
clear reason, set edge to blend so only the outer surroundings fade into the canvas. Inspect the top and bottom
edges as well as the sides. Keep framed when a distinct image panel improves the composition. This affects the
surroundings, leaving the subject intact.
Return one JSON object with exactly two keys: reason (a concrete Korean sentence) and revision. A revision
contains only boxes, styles, accent or edge in the same form as the supplied design. Each of boxes and styles,
if changed, must include headline, product_name and price. Boxes are [left, top, width, height] in whole
percentages. Their rectangles must not intersect. The end of one stacked box must be at or before the next
box's top. Use depth only for the value behind_subject and omit it to put type in front. Fix any reported
layout_error using the actual photograph. Return revision: null only when the layout works and layout_error is null."""

FINISHED_COMPOSITION_COMPARE = """The first poster is the current one, and the second tries to fix the stated
composition_issue. Compare the actual letter shapes, gaps, and their relationship to the photographed feature.
Use the visible ink bounds to locate the type. Count a change in position, size or color only when it makes
the reading path or hierarchy clearer.
When balanced_outer_space is present, image three keeps the first revision's type treatment with more modest
placement changes. Compare all three actual posters before choosing. Any reference images come afterward.
Start by looking at the full canvas in each poster. Use original_outer_space and revision_outer_space to inspect
the areas outside both the words and product. expanded_outer_space names the edges where the revision leaves
substantially more space. Moving text toward the product can reduce internal gaps while creating an empty band
on even one side. Prefer the existing layout or the moderate revision if the larger band serves no purpose.
An enlarged margin can work if it visibly supports the photographed action, guides the eye or gives an
important element emphasis. Explain that visible purpose when choosing a revision with more outer space.
Use product_bounds alongside both ink maps. When the headline moves from above the product to below it or
the reverse, explain what that move does to the entire reading path and the margin it leaves behind.
Choose revision only when the affected parts now read together more clearly or the issue is visibly solved.
If the request identifies several related problems, check each in both images. Keep the current poster if a
requested relationship still looks broken or if the change solves one by worsening another.
If the copy is readable in both, explain what happens to the advertised feature and title on the first glance.
When headline_priority is supplied, check whether the actual typography makes those words lead. Judge their
visible weight, size and color rather than assuming the first line is strongest. Check whether the revision
has weakened the title as a whole or opened up an unhelpful gap between the headline and the photo.
Return choice (original, revision, or balanced when image three is an alternative) and a concrete Korean reason
that describes what the images show."""

CROWDED_OFFER_COMPARE = """The posters follow option_order. They use the same photograph, headline, product
name, and exact price. Compare how the eye moves from the subject to the offer and whether both pieces of
information can be read together at phone-feed size. Prefer the option that gives them breathing room without
pulling attention away from the product, crowding a frame edge, or making a new detached text block. Keep
original if the larger type creates a worse composition. Return choice (one option_order key) and reason
(one concrete Korean sentence about what changes visibly)."""

CRITIQUE = """The first image is the previous finished ad, the second is the source product, and the rest are
design references. Identify the most consequential visible weakness of the ad that changing the photograph
could fix. Ground it in details you see. Propose one new visual device justified by a real detail of the product
or the campaign message. Learn from the references how the main subject and a decisive choice of scale, light,
material or type make an idea visible. A brighter background or a tidier product shot alone is insufficient.
Preserve product identity and visual presence. Borrow principles only. Return JSON
with 'weakness' (in Korean) and 'hypothesis' (an English art direction idea)."""

PHOTO_DIRECTION = """Write the English instruction that the image model will receive for this photograph.
The image is the source product. Make the design concept and any revision hypothesis visibly happen in the photograph.
Use the approved artwork box and copy positions to describe the subject and quiet space relative to the artwork.
Describe the camera framing, the exact visible effect, and how its light, material or gesture leads towards the
planned type. Translate the selected lighting into visible highlights and shadows on named parts of the product.
Keep the product prominent and recognizable, with its visible details, packaging and labels.
Describe the actual foreground item and its visible contents concretely. Preserve the visible kinds of items.
If the product name is broader than the photograph, do not fill in other items, varieties or components.
When there is a product demonstration, describe exactly what supports each object and the physical relationship
of the moving parts. Use surroundings and props only where they serve this particular image or the owner's
request. Follow the agreed crop. State only what the generated photograph should show. Include no ad text,
graphics, coordinates or layout markup in scene. Return JSON with 'scene', 'photo_cue' and 'companion'. Write photo_cue as one
short English sentence naming the visible relationship or effect that must be true for the scene to convey the
message. For a demonstration, name the relative position and direction of the moving parts, not just their
presence. Choose a moment that makes the intended action distinguishable from its reverse or an unrelated
gesture. Show the actual grip or support in frame and use a visible starting point and a separate destination,
contact or outcome for that distinction. The cue must be directly observable without reading the headline.
Use companion only for a distinct secondary object that an owner requirement in idea or product_facts, or
a headline with headline_locked=true, requires to communicate a pairing or situation. Return null for a
draft-only idea, decorative objects, mood alone, or parts involved in a
product demonstration. Otherwise give companion with object (short name), cue (which recognizable detail must be
visible), and box [left, top, width, height] in whole percentages of the photograph, relative to the artwork only.
In cue require only details needed for recognition and the owner's message. An optional color, shape or feature
of the styling is not a required detail unless the owner supplied it.
If the message needs what is inside or on a secondary object, require that visible detail in cue. A recognizable
container or tool alone does not show what it contains or carries.
Keep that box fully inside the frame with margin, separate from the product and away from the type. The object
must have room for its complete silhouette and enough detail to identify it in the full view. Adapt the camera
height slightly if a necessary opening or surface would be hidden. Keep optional foreground details subordinate.
When companion is required, scene is the first photograph: describe the product and setting with an empty
surface where the companion will be added later. Do not name or depict the companion in this first photograph.
Choose a box on a believable support surface with enough room for the object, its shadow, and blending edges."""

COMPANION_INSERTION = """Plan one local addition to this actual photograph. Preserve the product, lighting,
camera and existing surfaces. Find a genuinely empty area on a suitable support surface for the required object.
The proposed box is a suggestion. Adjust it if the photograph has a different usable surface. Keep away from
the product and rendered type. Use product_bounds as a measured envelope of the photographed product.
Leave at least four percent of the frame between the crop and each frame edge so the crop can blend.
The box must contain the whole object, its shadow and spare background for a
seamless blend. The object should be small enough to remain secondary. Return box [left, top, width, height]
in whole percentages of the photograph and instruction, an English instruction for only that crop. Describe the
object, its material and the actual local angle and placement on the existing surface. The image model sees
only the box you selected. Choose its size so an object occupying about three fifths of its longer axis looks
plausible at that depth relative to the product. Use a nearby surface at a similar depth when possible so the
two objects read as a pair. Its defining detail must be readable without zooming in. The
renderer specifies the object's size and centers it in the crop; omit size and position directions from
instruction. Mention no object that is outside the crop. If a second image is supplied, it is the rejected
addition. Use it and insertion_feedback to correct the position, scale or visible detail. Return box null
when no suitable surface exists."""

PHOTO_OBSERVATION = """Describe this single photograph as a viewer seeing it without any caption.
If someone handles an object, distinguish what holds or supports it, what visibly moves or is poised to move,
and what surface or object appears to receive it. Name the destination the picture actually suggests. A still
frame does not prove what happened before or after. For an unsupported tool, say it appears suspended rather
than inferring a hand outside the frame. If no action is depicted, report the most salient details
such as texture, shape, setting, light, and recognizable secondary objects. If an edge fragment is hard to
identify, describe what part is visible without guessing what the whole object is. Return one JSON object with
the exact key "observation" and no
other keys. Its value is a concrete Korean sentence about the visible subject and direction where relevant."""

PHOTO_MESSAGE = """Compare the exact Korean headline and message to the photograph without lettering and the
independent observation. Identify the one concrete thing the picture must communicate for this campaign.
For an action, check the visible support, what should move, and where it is visibly directed. Motion back into
the source does not show taking material out or putting it to use. An ambiguous direction or unsupported pose
does not show a specific advertised action just because the right tool is present.
For a visible scene, check the specific setting, pairing, color, texture or material effect the campaign draws
attention to. If the observer did not mention a secondary element, look for it in the actual image. Visible
forms must be recognizable in the full frame. A cut-off, indistinct shape does not establish a featured
pairing. A featured pairing needs a distinguishing visual cue for both things even without the headline.
Identify the actual visible features you used. Do not assume the identity of a blurry edge fragment from
the wording. Require only elements that carry the message.
A suggested time, mood, scent or subjective feeling need not be literally visible. Recognizable photographic
signals may convey the occasion or atmosphere without asserting a fact the camera cannot prove.
Return message_supported (boolean) and message_observation (one Korean sentence). If false, return repair_kind
(action or scene), corrected_photo_cue (one visible requirement), correction (a complete corrected scene), and
reposition (boolean). Set reposition to true only if the main product itself must change its framing or pose.
Moving, replacing, or adding a secondary object alone is reposition=false. Both instructions are in English.
For a scene correction, preserve the actual product
position, lighting and camera view where possible. Describe which specific visible element would convey the
message while keeping it secondary to the product. Return repair_box [left, top, width, height] as integer
percentages of the full photo for a local scene correction. It must cover the old shape, the whole new shape,
both shadows and enough surrounding material to blend. Exclude the product and its shadow. For a distinct
secondary object meant to read as a whole, return its description in secondary_object. Allow room for it away
from the frame edges, including if the old shape was clipped there. Leave secondary_object null for a light,
surface, or intentionally clipped detail. Return repair_box null if the requested change affects the whole scene.
For a local secondary object give patch_instruction, an English direction for this small area alone. Describe
its material, scale, angle, and contact with the existing surface. Omit descriptions of products or room features
that are outside the area, to avoid accidentally drawing them inside it.
For action corrections also give visible_support (the
actual visible contact that holds the moving item), and action_target (the distinct receiver or the visible
end of the action). Describe both as visible positions or contact in a still frame. Reuse any physically
valid grip already visible without inventing a different handling method."""

PHOTO_PATCH_CHECK = """Review this local photo correction at normal size. Image one is the original photo,
image two is the repaired photo, image three is the edited area up close. Verify the visible correction and
look for discontinuous materials, sharp patch edges, implausible light or shadow, and remnants at the frame edge.
If secondary_object is provided, there must be visible space between its entire silhouette and the photo edges.
Reject an object that is clipped or crowded against an edge even if the object can be identified. The corrected
element should stay subordinate to the main product while retaining a plausible scale for its position and
recognizable details at full-frame size. Return accepted (boolean) and observation
(a concrete Korean sentence about the result). When measure_object is true, also return object_box [left, top,
width, height] in integer percentages of image three, bounding the entire visible object without its shadow.
Return object_box null if the object cannot be identified."""

DRAFT_PHOTO_COPY = f"""Reconsider a first, unapproved campaign draft after seeing the actual photograph.
Image one is the source product and image two is the photograph to keep. Change the headline only if the
missing scene element was a creative choice in the draft and a distinct visible appeal can express the owner's
idea. An explicit pairing, use, occasion, offer or other required part of the owner's idea or product facts must
survive. Read requirements only from owner_idea and product_facts. The draft is an earlier creative suggestion.
Return {{"revision":null}} when the owner's intent would be lost. Otherwise return revision with headline
(natural Korean, at most {MAX_HEADLINE_LENGTH} characters including spaces), message (one Korean sentence), and
evidence (the specific visible product detail or context that supports the message). Avoid filler, product-name
repetition, unsupported claims, and ambiguous edge fragments. Choose the clearest distinctive product detail
that the viewer can actually see. Use the setting or mood as support. Avoid generic wording that would fit a
different item in the same scene. Write a line someone would put on a poster, not a literal catalog caption
explaining what is visible. Aim for 7 to 13 characters when natural. Where two lines would help, add first_line
as the exact initial words of headline up to a natural pause. Keep words that belong together on one line.
Use first_line null when there is no natural two-line break. Exclude missing scene elements from the revised
message as well as the headline. If visual_issue is supplied, address it without lowering the owner's intent."""

DRAFT_PHOTO_COPY_CHECK = """Independently judge the proposed headline and message using this photograph.
Check the concrete feature and the situation it depicts. Do not infer a missing feature from a shadow or an
indistinct fragment. The setting and light can suggest time or mood without proving it. Check that the proposed
copy serves the owner's idea and retains any explicitly required pairing, use, occasion, offer or product fact.
Only require facts relevant to the proposed change; offer, price and product name are already printed separately.
Return only JSON with exactly three keys: message_supported (boolean), intent_preserved (boolean), and
message_observation (one Korean sentence explaining the visual fit). Return both booleans even if one is false."""

PHOTO_CUE_CHECK = """Compare the original product (first image) and the rendered photograph without type
(second image). If has_approved_detail is true, the third image shows the selected source detail against
gray. The last four images are close views of the second image in order: top left, top right, bottom left
and bottom right. Independently check product identity in the second image. Compare the actual shape, proportions,
materials, visible layers, genuine packaging marks and product variants. With the whole product requested,
check that all sold pieces or visible variants are still represented, without duplicates or substitutions.
Use the approved detail when provided to decide what can be outside the frame. Allow the expected effects of
lighting, angle and an agreed close crop. Allow ordinary scene props that do not look like other sold products.
Report a concrete visible difference before marking identity_matches false.
Inspect all four close views for newly invented letters, numbers, logos or watermarks, including partial
or blurred marks on props touching the edges. Compare the original image before reporting a mark; preserve
lettering already visible in it. Always return stray_writing, with null if absent or an object with observation
(Korean) and box [left, top, width, height] in whole percentages of the full rendered photograph (image two).
Cover only the visible mark and keep its box within that image.
Separately identify a blocking_distraction only when a distinct, dispensable object competes with the sold
product by apparent size, placement, or strong contrast, or looks like an extra item being sold. Inspect the
whole scene. This includes unexpected objects partly behind a correctly rendered product. Keep scene-defining
surfaces, product containers, required tools, and parts that convey the approved idea or use. Keep identity_matches
about the original product itself when a separate background object is the problem. Return null when there is
no major distraction. Otherwise return blocking_distraction as an object with observation (a Korean sentence
naming the object, its location and why it competes) and scene (a complete positive English description of the
cleaned scene). In scene describe the material and lighting that should occupy the distracting area, keep the
product pose and the photo cue, and do not name the object to be omitted.
Return distractions as an array of up to two compact background objects that compete with the product or the
planned lettering area without supporting the message. Look for unintended duplicates or isolated cut-off
fragments with a strong edge or highlight. If two such objects are visible, box each one separately. Keep the
props that establish the setting or make the advertised use readable. Never list the sold item, its container,
supporting hands or tools needed for the action, cast shadows, or the supporting surface. Use an empty array when
there is no dispensable distraction. Each item has observation (Korean) and box in the same coordinates as
stray_writing. Describe the specific object and why it draws attention.
Verify photo_cue from the actual pixels in the second image when supplied. If required_companion is supplied,
also check its complete silhouette, its recognizable detail, and visible clearance to every frame edge. Set
visible to false if it is absent, clipped, hidden by the main product, crowds the frame edge, or has more visual
weight than the product. For a required companion, enforce specific placement only when it is part of owner_idea
or product_facts. Ignore proposed positions in art_direction when judging the actual framing. If photo_cue is null, set visible to true
and describe the visible arrangement without requiring a specific effect. For a gesture or motion, locate
the two relevant points and check their relative positions and direction. For light or material, locate the
specified surface and check the actual highlight, shadow, or transmission there. Visible parts alone do not
prove a relationship. If a planned action would physically miss its destination or appears suspended without
plausible support, report it. Return JSON with identity_matches and visible (separate booleans),
identity_observation and observation (one concrete Korean sentence for each finding), and correction
(one precise English photographic instruction if either finding is false or stray_writing is present,
otherwise null). In those cases,
also return reposition (boolean, true when the correction must change the relative position or orientation
of product parts, false when moving a secondary object alone). Correct the failed findings while keeping anything that already works. Describe a physically
complete arrangement when repositioning. Preserve the intended product scale and quiet area for lettering."""

PHOTO_CLEANUP_COMPARE = """Compare the same photograph before and after a local cleanup. Use distractions to
locate the changed areas. The first two images show the whole photograph before and after. Each following pair
shows one changed area before and after at the same scale. Judge the specific marked object there. A different
object outside that area may still belong to the scene. Choose cleaned only if the unintended objects are gone or less distracting and the
surface, shadows, and edges still look naturally continuous. Look for obvious smears, repeats, or ghost outlines.
Preserve the advertised subject, supporting objects, and the photographic evidence for the message. Keep original
if the edit removes something meaningful or introduces a new defect. Return choice (original or cleaned) and
reason (a concrete Korean sentence about the visible result)."""

FORMAT_STAGING = """The first image is the source product, the second the approved campaign, and the
third the isolated original product. Choose whether to physically move one already visible, independently
movable piece in the source to create a better reading path for target_canvas. Look at how the current
group runs across the source. If it sits on the wrong axis, consider pulling one real piece forward or
sliding it sideways on the same surface to give the group depth or direction. Preserve all product
variants, count, proportions and the campaign's visible motif. Do not propose moving an attached part
or changing size, camera angle, or the surface. Return {"move": "name the actual piece, new location and
desired visible geometry in one staging instruction"} or {"move": null} if the existing arrangement fits
or a safe move cannot resolve format_issue."""


def is_color(value):
    return isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value) is not None


def valid_headline_emphasis(emphasis, text):
    phrase = emphasis.get("phrase") if isinstance(emphasis, dict) else None
    return (isinstance(phrase, str) and bool(phrase) and phrase.strip() == phrase and phrase != text
            and text.count(phrase) == 1 and re.search(r"(?<!\S)" + re.escape(phrase) + r"(?!\S)", text) is not None
            and is_color(emphasis.get("color")))


def pixel_box(box, size):
    left, top, width, height = box
    canvas_w, canvas_h = size
    x, y = round(left * canvas_w / 100), round(top * canvas_h / 100)
    return (x, y, round((left + width) * canvas_w / 100) - x,
            round((top + height) * canvas_h / 100) - y)


def text_style(style, fonts, role):
    font, align, color, size = (style.get(key) for key in ("font", "align", "color", "size"))
    if (font not in tuple(fonts) or align not in ("left", "center", "right") or not is_color(color)
            or type(size) not in (int, float) or not 2 <= size <= 20):
        raise ValueError(f"{role} 글자 스타일 값이 올바르지 않습니다")
    return {"font": font, "align": align, "color": color}


def headline_breaks(text):
    words = text.split()
    return [{"break_after": index, "first_line": " ".join(words[:index]), "second_line": " ".join(words[index:])}
            for index in range(1, len(words))]


def headline_lines(layer, parent_style, canvas_size, fonts):
    styles = parent_style["lines"]
    if (not isinstance(styles, list) or len(styles) != 2 or "break_after" not in layer
            or any(not isinstance(style, dict)
                   or style.keys() - {"box", "size", "font", "align", "color", "depth", "text", "break_after"}
                   for style in styles)):
        raise ValueError("나눠 배치할 제목에는 줄바꿈과 두 줄의 위치 및 스타일을 지정해야 합니다")
    split = layer["break_after"]
    if any(type(style.get("break_after", split)) is not int or style.get("break_after", split) != split
           for style in styles):
        raise ValueError("제목 줄에 반복된 줄바꿈 위치가 제목 전체의 줄바꿈과 다릅니다")
    boxes = {str(index): style.get("box") for index, style in enumerate(styles, 1)}
    validate_boxes(boxes, tuple(boxes))
    if boxes["1"][1] + boxes["1"][3] > boxes["2"][1]:
        raise ValueError("제목의 첫 줄은 둘째 줄보다 위에 있어야 합니다")
    text = layer["text"]
    words = list(re.finditer(r"\S+", text))
    parts = [(0, words[split - 1].end()), (words[split].start(), len(text))]
    output = []
    for index, ((start, end), style) in enumerate(zip(parts, styles, strict=True), 1):
        line_text = text[start:end]
        if style.get("text", line_text) != line_text:
            raise ValueError("제목 줄의 문구가 확정된 원문과 다릅니다")
        x, y, w, h = pixel_box(boxes[str(index)], canvas_size)
        if not (layer["x"] <= x < x + w <= layer["x"] + layer["w"]
                and layer["y"] <= y < y + h <= layer["y"] + layer["h"]):
            raise ValueError("제목 각 줄은 제목 전체 영역 안에 들어가야 합니다")
        visual = parent_style | style
        child = text_style(visual, fonts, "제목 줄") | {
            "text": line_text, "role": f"headline_line_{index}", "x": x, "y": y, "w": w, "h": h,
            "file": f"text_headline_{index}.png", "max_lines": 1,
            "max_font_size": round(visual["size"] * canvas_size[0] / 100), "min_font_size": layer["min_font_size"],
        }
        depth = style.get("depth", layer.get("depth"))
        if depth not in (None, "front", "behind_subject"):
            raise ValueError("제목 줄의 깊이는 제품 앞 또는 뒤여야 합니다")
        if depth == "behind_subject":
            child["depth"] = depth
        if emphasis := layer.get("emphasis"):
            marked = text.index(emphasis["phrase"])
            first, last = max(start, marked), min(end, marked + len(emphasis["phrase"]))
            if first < last:
                child["emphasis"] = {"phrase": text[first:last], "color": emphasis["color"], "start": first - start}
        output.append(child)
    return output


def layout_texts(answer, copy, canvas_size, fonts):
    boxes = validate_boxes(answer.get("boxes"), TEXT_ROLES)
    styles = answer.get("styles")
    if not isinstance(styles, dict) or styles.keys() != boxes.keys():
        raise ValueError("문구마다 글꼴, 정렬, 색과 크기가 필요합니다")
    width = canvas_size[0]
    texts = []
    for role, text in copy.items():
        x, y, w, h = pixel_box(boxes[role], canvas_size)
        style = styles[role]
        if not isinstance(style, dict):
            raise ValueError(f"{role} 글자 스타일이 빠졌습니다")
        visual = text_style(style, fonts, role)
        size, max_lines, break_after = (style.get(key) for key in ("size", "max_lines", "break_after"))
        if type(max_lines) is not int or not 1 <= max_lines <= (1 if role == "price" else 2):
            raise ValueError(f"{role} 글자 스타일 값이 올바르지 않습니다")
        layer = visual | {"role": role, "text": text,
                 "x": x, "y": y, "w": w, "h": h, "file": f"text_{role}.png", "max_lines": max_lines,
                 "max_font_size": round(size * width / 100), "min_font_size": minimum_font_size(width)}
        depth = style.get("depth")
        if depth not in (None, "front"):
            if role != "headline" or depth != "behind_subject":
                raise ValueError("제품 뒤에 걸 수 있는 글자는 제목뿐입니다")
            layer["depth"] = depth
        if break_after is not None:
            if max_lines != 2 or type(break_after) is not int or not 0 < break_after < len(text.split()):
                raise ValueError(f"{role} 줄바꿈 위치가 올바르지 않습니다")
            layer["break_after"] = break_after
        emphasis = style.get("emphasis")
        if emphasis is not None:
            if role != "headline" or not valid_headline_emphasis(emphasis, text):
                raise ValueError("강조는 제목에 한 번 나오는 일부 어절과 올바른 색을 지정해야 합니다")
            layer["emphasis"] = {key: emphasis[key] for key in ("phrase", "color")}
        if style.get("lines") is not None:
            if role != "headline":
                raise ValueError("줄마다 따로 배치할 수 있는 문구는 제목입니다")
            layer["lines"] = headline_lines(layer, style, canvas_size, fonts)
            del layer["file"]
        texts.append(layer)
    return texts


def design_layers(answer, copy, size, fonts, folder):
    required_text(answer, "concept", "포스터 구도")
    required_text(answer, "art_direction", "포스터 구도")
    accent = answer.get("accent")
    if (answer.get("focus") not in LAYOUT_ROLES or answer.get("crop") not in ("preserve", "intentional")
            or answer.get("edge") not in ("blend", "framed")
            or not is_color(answer.get("background"))):
        raise ValueError("포스터 구도에 강조점, 제품 잘림 정책과 배경색이 필요합니다")
    validate_boxes({"artwork": answer.get("artwork_box")}, ("artwork",))
    if "subject_box" in answer:
        validate_boxes({"subject": answer["subject_box"]}, ("subject",))
    if answer.get("source_crop") is not None:
        if answer["crop"] != "intentional":
            raise ValueError("원본 일부를 확대하려면 의도한 잘림으로 지정하세요")
        validate_boxes({"source_crop": answer["source_crop"]}, ("source_crop",))
    x, y, w, h = pixel_box(answer["artwork_box"], size)
    if min(answer["artwork_box"][2:]) < 35:
        raise ValueError("상품 사진을 보여 줄 영역이 너무 작습니다")
    background = Image.new("RGB", size, answer["background"])
    graphic = None
    if accent is not None:
        if (not isinstance(accent, dict) or accent.get("shape") not in ("rectangle", "ellipse")
                or accent.get("layer", "background") not in ("background", "overlay", "behind_subject")
                or not is_color(accent.get("color"))):
            raise ValueError("색면의 형태와 색이 올바르지 않습니다")
        validate_boxes({"accent": accent.get("box")}, ("accent",))
        ax, ay, aw, ah = pixel_box(accent["box"], size)
        overlay = accent.get("layer") in ("overlay", "behind_subject")
        if not overlay and (accent["color"].lower() == answer["background"].lower()
                            or (x <= ax and y <= ay and x + w >= ax + aw and y + h >= ay + ah)):
            raise ValueError("배경 색면은 바탕과 구분되고 사진 밖에서 보여야 합니다")
        target = Image.new("RGBA", (aw, ah)) if overlay else background
        left, top = (0, 0) if overlay else (ax, ay)
        draw = ImageDraw.Draw(target)
        shape = draw.rectangle if accent["shape"] == "rectangle" else draw.ellipse
        shape((left, top, left + aw - 1, top + ah - 1), fill=accent["color"])
        if overlay:
            graphic = {"file": "graphic.png", "x": ax, "y": ay, "w": aw, "h": ah}
            if accent["layer"] == "behind_subject":
                graphic["depth"] = "behind_subject"
            target.save(folder / graphic["file"])
    texts = layout_texts(answer, copy, size, fonts)
    render_texts(texts, size, folder)
    background.save(folder / "background.png")
    layout = {"size": list(size), "background": {"file": "background.png"}, "texts": texts,
              "artwork": {"file": "scene.png", "x": x, "y": y, "w": w, "h": h}}
    if graphic:
        layout["graphic"] = graphic
    return layout


def extract_subject(photo, mask_model, threshold):
    from ..photo_retouch.product_mask import product_mask

    mask = product_mask(mask_model, to_float(photo))
    subject = Image.fromarray(mask > threshold)
    product = cutout_product(photo, mask, threshold) if subject.getbbox() else None
    return product, subject


def crop_subject(product, design):
    if product is None or design.get("source_crop") is None:
        return product
    x, y, width, height = pixel_box(design["source_crop"], product.size)
    detail = product.crop((x, y, x + width, y + height))
    if not detail.getchannel("A").getbbox():
        raise ValueError("확대할 부분에 원본 상품이 없습니다")
    return detail


def photo_review_details(scene):
    width, height = scene.size
    crop_width, crop_height = ceil(5 * width / 8), ceil(5 * height / 8)
    return [to_float(scene.crop((x, y, x + crop_width, y + crop_height)))
            for y in (0, height - crop_height) for x in (0, width - crop_width)]


def remove_photo_writing(scene, foreground, mark):
    from ..photo_retouch.cleanup import load_inpaint_model, remove_distractions
    from ..photo_retouch.settings import load_settings as retouch_settings

    x, y, width, height = validate_boxes({"writing": mark.get("box")}, ("writing",))["writing"]
    rule = retouch_settings()["cleanup"]
    if width * height > 10000 * rule["max_box_share"]:
        return None
    try:
        model = load_inpaint_model()
    except FileNotFoundError:
        return None
    region = {"box": [10 * value for value in (x, y, x + width, y + height)], "label": "writing"}
    cleaned, removed = remove_distractions(model, to_float(scene), [region], foreground, rule)
    return Image.fromarray(np.rint(255 * cleaned).astype(np.uint8)) if removed else None


def fit_companion_patch(patch, mask):
    box = Image.fromarray(mask).getbbox()
    if box is None:
        return None
    x, y, right, bottom = box
    width, height = patch.size
    if x == 0 or y == 0 or right == width or bottom == height:
        return None
    object_width, object_height = right - x, bottom - y
    scale = min(2, 0.62 * max(patch.size) / max(object_width, object_height),
                0.72 * width / object_width, 0.72 * height / object_height)
    if scale < 1.12:
        return None

    offset_x = (width - (x + right) * scale) / 2
    offset_y = (height - (y + bottom) * scale) / 2
    # Feather against the original surface where the transformed source has real pixels.
    left, top = max(0, ceil(offset_x)), max(0, ceil(offset_y))
    end_x = min(width, floor(width * scale + offset_x))
    end_y = min(height, floor(height * scale + offset_y))
    margin = min(end_x - left, end_y - top) / 8
    if min(x * scale + offset_x - left, y * scale + offset_y - top,
           end_x - right * scale - offset_x, end_y - bottom * scale - offset_y) < margin:
        return None
    size = (end_x - left, end_y - top)
    transform = (1 / scale, 0, (left - offset_x) / scale, 0, 1 / scale, (top - offset_y) / scale)
    resized = patch.transform(size, Image.Transform.AFFINE, transform, Image.Resampling.BICUBIC)
    silhouette = Image.fromarray(mask).transform(size, Image.Transform.AFFINE, transform, Image.Resampling.NEAREST)
    return resized, np.asarray(silhouette), (left, top, end_x, end_y)


def match_companion_surface(patch, surface, mask):
    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        return patch
    radius = min(patch.size)
    distance = distance_transform_edt(~mask)
    background = distance > 0.15 * radius
    if background.mean() < 0.15:
        return patch
    edited, original = to_float(patch), to_float(surface)
    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    difference = np.log((edited @ luma + 0.02) / (original @ luma + 0.02))
    y, x = np.indices(mask.shape, dtype=np.float32)
    plane = np.stack((np.ones_like(x), x / patch.width, y / patch.height), axis=-1)
    samples = difference[background]
    gain = plane @ np.linalg.lstsq(plane[background], samples, rcond=None)[0]
    low, high = np.percentile(samples, (5, 95))
    balanced = edited * np.exp(-gain.clip(low, high))[..., None]

    # Preserve the original surface beyond the silhouette and its nearby contact shadow.
    spill = (balanced @ luma - original @ luma - 0.025).clip(0, None)
    balanced -= (spill * (distance / (0.035 * radius)).clip(0, 1))[..., None]
    edge = ((0.33 * radius - distance) / (0.27 * radius)).clip(0, 1)
    alpha = (edge ** 2 * (3 - 2 * edge))[..., None]
    blended = balanced * alpha + original * (1 - alpha)
    return Image.fromarray(np.rint(255 * blended.clip(0, 1)).astype(np.uint8))


def repair_scene_patch(model, scene, bounds, instruction, generation, *, secondary_object=None, foreground=None,
                       companion_mask=None):
    x, y, right, bottom = bounds
    margin = 0 if companion_mask is not None else min(right - x, bottom - y) // 2
    left, top, end_x, end_y = padded_box(bounds, margin, scene.size)
    if secondary_object and foreground is not None and foreground[top:end_y, left:end_x].any():
        left, top, end_x, end_y = bounds
    window = scene.crop((left, top, end_x, end_y))
    context = ImageOps.contain(window, (576, 576), Image.Resampling.LANCZOS)
    patch_box = (x - left, y - top, right - left, bottom - top)
    def percentages(box):
        return tuple(round(100 * edge / length) for edge, length in zip(box, window.size * 2, strict=True))

    region = percentages(patch_box)
    placement = ""
    if secondary_object:
        frame_gap = round(min(scene.size) * 0.04)
        feather = min(right - x, bottom - y) // 12
        inner = (max(x + feather, frame_gap), max(y + feather, frame_gap),
                 min(right - feather, scene.width - frame_gap), min(bottom - feather, scene.height - frame_gap))
        x0, y0, x1, y1 = inner
        if x0 >= x1 or y0 >= y1:
            raise ValueError("소품을 가장자리에서 띄워 배치할 공간이 부족합니다")
        object_area = percentages((x0 - left, y0 - top, x1 - left, y1 - top))
        placement = (f"Fit the complete silhouette inside {object_area} percent of this view. "
                     f"Intended object: {secondary_object}\n"
                     "Keep the existing surface continuous around "
                     "the object. Add no other items, repeated products, new surface edges or borders. ")
        if companion_mask is not None:
            placement += ("Center the object. Its longer axis should fill about three fifths of that axis of this crop. "
                          "Leave visible surface on all sides, including around its contact shadow. ")
        else:
            placement += "Reduce its scale or move it inward as needed. "
    prompt = (f"Make a local commercial-photo retouch within {region} in percent (left, top, right, bottom). "
              f"{instruction} {placement}Keep the surrounding material, camera view, depth of field and light continuous. "
              "Keep the entire replacement and its shadow inside the edit area. Add no lettering or graphics.")
    candidate = model.edit(context, prompt, generation["seed"], generation["steps"], size=context.size)
    patch = candidate.convert("RGB").resize(window.size, Image.Resampling.LANCZOS).crop(patch_box)
    if companion_mask is not None:
        mask = companion_mask(patch)
        if (fitted := fit_companion_patch(patch, mask)) is not None:
            patch, mask, (left, top, end_x, end_y) = fitted
            bounds = (x + left, y + top, x + end_x, y + end_y)
        patch = match_companion_surface(patch, scene.crop(bounds), mask)
    repaired = scene.copy()
    paste_region(repaired, patch, bounds)
    return repaired


def clean_photo_distractions(ai, scene, foreground, photo_regions, distractions, message):
    from scipy.ndimage import binary_dilation
    from ..photo_retouch.cleanup import box_pixels, load_inpaint_model, remove_distractions
    from ..photo_retouch.settings import load_settings as retouch_settings

    if not isinstance(distractions, list) or not 1 <= len(distractions) <= 2:
        return scene, None
    rule = retouch_settings()["cleanup"]
    protected = foreground | photo_region_mask(photo_regions, scene.size)
    keep = binary_dilation(protected, iterations=rule["subject_margin_px"])
    width, height = scene.size
    repairs = []
    for item in distractions:
        if not isinstance(item, dict):
            continue
        try:
            label = required_text(item, "observation", "사진의 배경 소품 검토")
            x, y, w, h = validate_boxes({"object": item.get("box")}, ("object",))["object"]
        except ValueError:
            continue
        box = [10 * value for value in (x, y, x + w, y + h)]
        left, top, right, bottom = box_pixels(box, width, height, rule["box_padding"])
        if (right - left) * (bottom - top) > rule["max_box_share"] * width * height:
            continue
        if not keep[top:bottom, left:right].any():
            repairs.append({"box": box, "label": label})
    if not repairs:
        return scene, None
    try:
        model = load_inpaint_model()
    except FileNotFoundError:
        return scene, None
    original = to_float(scene)
    cleaned, removed = remove_distractions(model, original, repairs, protected, rule)
    if not removed:
        return scene, None
    details = []
    for repair in repairs:
        left, top, right, bottom = box_pixels(repair["box"], width, height, 0.5)
        details.extend((original[top:bottom, left:right], cleaned[top:bottom, left:right]))
    review = ask_stage(ai, "ad_photo_cleanup_compare", PHOTO_CLEANUP_COMPARE,
                       {"distractions": repairs, "message": message}, [original, cleaned, *details])
    reason = review.get("reason")
    if review.get("choice") != "cleaned" or not isinstance(reason, str) or not reason.strip():
        return scene, None
    return Image.fromarray(np.rint(255 * cleaned).astype(np.uint8)), {"removed": removed, "reason": reason}


def blocking_distraction(check):
    obstruction = check.get("blocking_distraction")
    if not isinstance(obstruction, dict):
        return None
    try:
        observation = required_text(obstruction, "observation", "촬영 배경 검토")
        scene = required_text(obstruction, "scene", "촬영 배경 수정")
    except ValueError:
        return None
    return observation, scene


def reconcile_draft_headline(ai, fields, idea, brief, original, picture, finding):
    context = {"product_facts": fields, "owner_idea": idea}
    draft = {key: brief[key] for key in ("headline", "message")}
    inputs = context | {"draft": draft, "observed_scene": finding["observed_scene"],
                        "mismatch": finding["message_observation"]}
    for _ in range(2):
        revision = ask_stage(ai, "ad_draft_photo_copy", DRAFT_PHOTO_COPY, inputs, [original, picture]).get("revision")
        if not isinstance(revision, dict):
            return None
        try:
            text = {key: required_text(revision, key, "사진에 맞춘 문구") for key in ("headline", "message", "evidence")}
        except ValueError:
            return None
        text["headline"] = " ".join(text["headline"].split())
        if len(text["headline"]) > MAX_HEADLINE_LENGTH or text["headline"] == brief["headline"]:
            return None
        if fields["product_name"] in text["headline"]:
            inputs.update(previous_revision=text, visual_issue="상품명이 별도 줄에 있으므로 제목에서 전체 상품명을 반복하지 마세요.")
            continue
        check = ask_stage(ai, "ad_draft_photo_copy_check", DRAFT_PHOTO_COPY_CHECK, text | context, [picture])
        observation = check.get("message_observation")
        if check.get("intent_preserved") is not True or not isinstance(observation, str) or not observation.strip():
            return None
        supported = check.get("message_supported")
        if supported is True:
            line_break = next((line["break_after"] for line in headline_breaks(text["headline"])
                               if line["first_line"] == revision.get("first_line")), None)
            return brief | text | {"scene": finding["observed_scene"]}, observation.strip(), line_break
        if supported is not False:
            return None
        inputs.update(previous_revision=text, visual_issue=observation.strip())
    return None


def planned_companion(direction, artwork, subject, canvas_size, lettering):
    plan = direction.get("companion")
    if plan is None:
        return None
    if not isinstance(plan, dict):
        raise ValueError("촬영 소품의 배치가 올바르지 않습니다")
    item = {key: required_text(plan, key, "함께 보이는 소품") for key in ("object", "cue")}
    box = validate_boxes({"companion": plan.get("box")}, ("companion",))["companion"]
    size = (artwork["w"], artwork["h"])
    _, _, width, height = pixel_box(box, size)
    gap = round(min(size) * 0.04)
    inset_x, inset_y = (ceil(100 * gap / side) for side in size)
    x, y, w, h = box
    if min(width, height) < 2 * gap or w + 2 * inset_x > 100 or h + 2 * inset_y > 100:
        raise ValueError("촬영 소품을 알아볼 수 있도록 가장자리에서 띄울 공간이 필요합니다")
    box = [max(inset_x, min(x, 100 - w - inset_x)), max(inset_y, min(y, 100 - h - inset_y)), w, h]
    left, top, width, height = pixel_box(box, size)
    if subject is not None and subject.getchannel("A").crop((left, top, left + width, top + height)).getbbox():
        raise ValueError("촬영 소품과 상품의 실루엣이 겹칩니다")
    x, y = left + artwork["x"], top + artwork["y"]
    for text_left, text_top, text_right, text_bottom in lettering.values():
        if (max(x * 100, text_left * canvas_size[0]) < min((x + width) * 100, text_right * canvas_size[0])
                and max(y * 100, text_top * canvas_size[1]) < min((y + height) * 100, text_bottom * canvas_size[1])):
            raise ValueError("촬영 소품의 위치가 실제 글자와 겹칩니다")
    return item | {"box": box}


def plan_companion_insertion(ai, scene, companion, foreground, layout, folder, feedback, rejected=None):
    subject = Image.new("RGBA", scene.size)
    subject.putalpha(Image.fromarray(foreground).convert("L"))
    x, y, right, bottom = subject.getbbox()
    product_bounds = [round(100 * value / side) for value, side in
                      zip((x, y, right, bottom), scene.size * 2, strict=True)]
    lettering = rendered_text_bounds(layout, folder)
    inputs = {"companion": companion, "artwork": layout["artwork"], "canvas_size": layout["size"],
              "product_bounds": product_bounds, "rendered_text_bounds": lettering, "insertion_feedback": feedback}
    images = [to_float(scene)]
    if rejected is not None:
        images.append(to_float(rejected))
    for _ in range(2):
        placement = ask_stage(ai, "ad_companion_insertion", COMPANION_INSERTION, inputs, images)
        if "box" in placement and placement["box"] is None:
            raise ValueError("완성 사진에 필수 소품을 자연스럽게 놓을 빈 면이 없습니다")
        try:
            planned = planned_companion({"companion": companion | {"box": placement.get("box")}},
                                        layout["artwork"], subject, layout["size"], lettering)
            instruction = required_text(placement, "instruction", "촬영 소품 부분 편집")
            left, top, width, height = pixel_box(planned["box"], scene.size)
            return (left, top, left + width, top + height), f"{instruction} Required visible detail: {planned['cue']}", planned
        except ValueError as error:
            inputs.update(previous_box=placement.get("box"), layout_error=str(error))
    raise ValueError(inputs["layout_error"])


def companion_has_visible_scale(box):
    try:
        accepted = validate_boxes({"object": box}, ("object",))["object"]
    except ValueError:
        return False
    return max(accepted[2:]) >= 40


def choose_source_framing(ai, product, inputs, images):
    whole = on_gray(product)
    answer = ask_stage(ai, "ad_source_framing", SOURCE_FRAMING, inputs, [to_float(whole), *images])
    crop = answer.get("source_crop")
    reason = answer.get("reason")
    if "source_crop" not in answer or not isinstance(reason, str) or not reason.strip():
        return {"source_crop": None, "reason": "추천 구도를 확인하지 못해 원본 범위를 유지합니다"}
    if crop is None or crop == [0, 0, 100, 100]:
        return {"source_crop": None, "reason": reason}
    try:
        validate_boxes({"source_crop": crop}, ("source_crop",))
        detail = crop_subject(product, answer)
    except ValueError as error:
        return {"source_crop": None, "reason": f"{error}. 원본 범위를 유지합니다"}
    frames = [to_float(ImageOps.pad(view, (512, 512), Image.Resampling.LANCZOS, color="#808080"))
              for view in (whole, on_gray(detail))]
    observed = ask_stage(ai, "ad_source_detail_observation", SOURCE_DETAIL_OBSERVATION, {}, [frames[1]])
    observation = observed.get("observation")
    fragments = observed.get("competing_fragments")
    if not isinstance(observation, str) or not observation.strip() or type(fragments) is not bool:
        return {"source_crop": None, "reason": "확대본의 피사체를 확인하지 못해 원본 범위를 유지합니다"}
    if fragments:
        return {"source_crop": None, "reason": observation}
    review = ask_stage(ai, "ad_source_framing_check", SOURCE_FRAMING_CHECK,
                       inputs | {"detail_observation": observation}, [*frames, images[0]])
    review_reason = review.get("reason")
    if review.get("choice") not in ("whole", "detail") or not isinstance(review_reason, str) or not review_reason.strip():
        return {"source_crop": None, "reason": "확대본의 비교 결과가 없어 원본 범위를 유지합니다"}
    identity = review.get("detail_preserves_identity")
    if review["choice"] == "whole" or identity is False:
        return {"source_crop": None, "reason": review_reason}
    if identity is not True:
        return {"source_crop": None, "reason": "확대본에서 상품을 식별할 수 있는지 확인하지 못해 원본 범위를 유지합니다"}
    return {"source_crop": crop, "reason": review_reason}


def frame_subject(product, picture_size, subject_box, source_crop=None):
    x, y, width, height = pixel_box(subject_box, picture_size)
    contained = ImageOps.contain(product, (width, height), Image.Resampling.LANCZOS)
    extra_width, extra_height = width - contained.width, height - contained.height
    offset_x, offset_y = extra_width // 2, extra_height // 2
    if source_crop:
        left, top, crop_width, crop_height = source_crop
        if left > 0 and left + crop_width == 100:
            offset_x = 0
        elif left == 0 and crop_width < 100:
            offset_x = extra_width
        if top > 0 and top + crop_height == 100:
            offset_y = 0
        elif top == 0 and crop_height < 100:
            offset_y = extra_height
    frame = Image.new("RGBA", picture_size)
    frame.alpha_composite(contained, (x + offset_x, y + offset_y))
    return frame


def product_reference(photo, product, picture_size, subject_box, source_crop=None):
    if product is None:
        return photo
    if subject_box:
        reference = Image.new("RGB", picture_size, "#808080")
        framed = frame_subject(product, picture_size, subject_box, source_crop)
        reference.paste(framed, mask=framed.getchannel("A"))
        return reference
    margin = round(max(product.size) / 10)
    reference = Image.new("RGB", (product.width + 2 * margin, product.height + 2 * margin), "#808080")
    reference.paste(product, (margin, margin), product)
    return reference


def composition_preview(product, design, layout, folder):
    artwork = layout["artwork"]
    size = (artwork["w"], artwork["h"])
    subject = frame_subject(crop_subject(product, design), size, design.get("subject_box", [8, 8, 84, 84]),
                            design.get("source_crop"))
    image = Image.new("RGBA", size, design["background"])
    image.alpha_composite(subject)
    image.save(folder / artwork["file"])
    foreground = np.asarray(subject.getchannel("A")) > 128
    Image.fromarray(foreground).save(folder / "foreground.png")
    return compose(layout, folder), foreground


def touches_photo_edge(mask):
    return mask[[0, -1], :].any() or mask[:, [0, -1]].any()


def continuous_offer_scene(design, ink):
    if (design["edge"] != "blend" or design["crop"] != "preserve"
            or not {"product_name", "price"} <= ink.keys()):
        return None
    x, y, width, height = design["artwork_box"]
    offer = [ink["product_name"], ink["price"]]
    separated = (max(box[2] for box in offer) <= x or min(box[0] for box in offer) >= x + width
                 or max(box[3] for box in offer) <= y or min(box[1] for box in offer) >= y + height)
    if not separated:
        return None
    sx, sy, sw, sh = design.get("subject_box", [8, 8, 84, 84])
    left, right = (round(x + width * edge / 100) for edge in (sx, sx + sw))
    top, bottom = (round(y + height * edge / 100) for edge in (sy, sy + sh))
    accent = design.get("accent")
    if accent and accent.get("layer", "background") == "background":
        ax, ay, aw, ah = accent["box"]
        covers_offer = all(ax <= l and ay <= t and r <= ax + aw and b <= ay + ah for l, t, r, b in offer)
        accent = None if covers_offer else accent | {"layer": "overlay"}
    return design | {"artwork_box": [0, 0, 100, 100], "edge": "framed",
                     "subject_box": [left, top, right - left, bottom - top], "accent": accent}


def review_offer_scene(ai, product, design, layout, folder, inputs, images):
    alternative = continuous_offer_scene(design, rendered_text_bounds(layout, folder))
    if alternative is None:
        return None
    original = compose(layout, folder)
    size = tuple(layout["size"])
    art = layout["artwork"]
    ImageDraw.Draw(original).rectangle((art["x"], art["y"], art["x"] + art["w"] - 1, art["y"] + art["h"] - 1),
                                     outline="#a4a4a4", width=max(1, min(size) // 256))
    with TemporaryDirectory(dir=folder) as temporary:
        candidate = Path(temporary)
        try:
            updated = design_layers(alternative, inputs["copy"], size, inputs["fonts"], candidate)
            rendered, foreground = composition_preview(product, alternative, updated, candidate)
            check_text_overlap(foreground, updated, candidate)
        except ValueError:
            return None
        review = ask_stage(ai, "ad_offer_scene", OFFER_SCENE_COMPARE,
                           {"copy": inputs["copy"], "brief": inputs["brief"],
                            "original_photo_bounds": design["artwork_box"]},
                           [to_float(original), to_float(rendered), *images])
        reason = review.get("reason")
        if review.get("choice") != "continuous" or not isinstance(reason, str) or not reason.strip():
            return None
        rendered.save(candidate / "composition.png")
        for path in candidate.iterdir():
            path.replace(folder / path.name)
    return alternative, updated, reason


def measured_format_composition(ai, design, layout, folder, foreground, first, inputs):
    from .format_layout import format_measurements, measured_styles, position_options, preview_placement

    size = tuple(layout["size"])
    artwork = layout["artwork"]
    x, y, w, h = (artwork[key] for key in ("x", "y", "w", "h"))
    mask = np.zeros((size[1], size[0]), bool)
    mask[y:y + h, x:x + w] = foreground
    box = Image.fromarray(mask).getbbox()
    if box is None:
        return None
    bounds = [round(100 * edge / limit) for edge, limit in zip(box, size * 2, strict=True)]
    measured = layout | {"design": design}
    options = position_options(measured, size, mask, bounds, format_measurements(measured, size))
    if not options:
        return None
    artwork_only = compose(layout | {"texts": [], "graphic": None}, folder)
    previews = [to_float(preview_placement(artwork_only, layout, folder, plan)) for plan in options.values()]
    answer = ask_stage(ai, "ad_format_measured_layout",
                       "Image 1 is the rejected native layout. The remaining images use the same actual subject "
                       "and exact legible words, placed near its silhouette in option_order. Choose the one that "
                       "visibly resolves format_issue and layout_error. Consider both the lines of the headline "
                       "and the product name and price as groups. Check the full frame on the long axis. "
                       "Return placement as one option key or null if none works, and reason in Korean.",
                       {"option_order": list(options), "format_issue": inputs["revision_feedback"]["format_issue"],
                        "layout_error": inputs.get("layout_error"), "copy": inputs["copy"]},
                       [to_float(first), *previews])
    placement = answer.get("placement")
    if not isinstance(placement, str) or placement not in options:
        return None
    plan = options[placement]
    revised = design | {"boxes": plan["boxes"], "styles": measured_styles(design, plan), "accent": None}
    return revised, required_text(answer, "reason", "게시 규격의 실제 글자 배치")


def review_composition_plan(ai, product, design, layout, folder, inputs, images):
    first, foreground = composition_preview(product, design, layout, folder)
    photo_input = [to_float(first), *images]
    review_input = inputs | {"design": design, "rendered_text_bounds": rendered_text_bounds(layout, folder)}
    task = BLOCKING + "\n" + DESIGN_SCHEMA
    format_issue = inputs.get("revision_feedback", {}).get("format_issue")
    has_failed_resize = inputs.get("has_failed_resize")
    if format_issue:
        task += ("\nThe draft is for the target canvas_size. Check the visible geometry against format_issue. "
                 "If the same gap or disconnected offer remains, revise the shared geometry, including the "
                 "first focal point and negative space. Future lighting alone is not a geometry change. "
                 "When crop is preserve, keep a clear edge around the actual silhouette. Increase its span "
                 "along the long axis using composition, not stretching or squeezing every edge. "
                 "When returning revision null, also return ready true only if the actual draft resolves the issue.")
        if has_failed_resize:
            task += (" The last image is the failed resize. Compare it to the first image, which shows the new "
                     "placement before photography. In the reason, name how the product, copy and any defining "
                     "scene element will occupy the previously unused part of the frame.")
    if "layout_device" in inputs.get("revision_feedback", {}):
        task += ("\nTest how revision_feedback.layout_device appears in the rendered draft. "
                 "If it only exists in the written plan, move or resize the artwork, lettering or graphic "
                 "until the visible relationship is present. Judge the intended effect at a small size.")
    if "source_framing" in inputs:
        task += "\nKeep the chosen source_framing. Adjust placement, scale and type around that photographed view."

    def check_draft(mask, plan, candidate_layout, candidate_folder):
        check_text_overlap(mask, candidate_layout, candidate_folder)
        if format_issue:
            if plan["crop"] == "preserve" and touches_photo_edge(mask):
                raise ValueError("전체를 보여 줄 상품이 사진 가장자리에 닿습니다")
            check_offer_spacing(rendered_text_bounds(candidate_layout, candidate_folder), candidate_layout["size"])

    editable = {"concept", "focus", "background", "accent", "artwork_box", "subject_box", "crop", "source_crop",
                "edge", "boxes", "styles"}
    if "source_framing" in inputs:
        editable -= {"crop", "source_crop"}
    original_valid = True
    try:
        check_draft(foreground, design, layout, folder)
    except ValueError as error:
        original_valid = False
        review_input["layout_error"] = str(error)

    kept_reason = None
    for _ in range(2):
        review = ask_stage(ai, "ad_composition", task, review_input, photo_input)
        revision = review.get("revision")
        if not isinstance(revision, dict):
            if format_issue:
                observation = required_text(review, "reason", "게시 규격 촬영 전 검토")
                if review.get("ready") is not True or not original_valid:
                    review_input["layout_error"] = observation
                    continue
                kept_reason = observation
            break
        changed = design | {key: revision[key] for key in editable & revision.keys()}
        if changed == design:
            if format_issue:
                review_input["layout_error"] = "현재 배열에서 달라진 항목이 없습니다"
                continue
            break
        with TemporaryDirectory(dir=folder) as temporary:
            candidate = Path(temporary)
            try:
                revised = design_layers(changed, inputs["copy"], tuple(layout["size"]), inputs["fonts"], candidate)
                rendered, proposed_foreground = composition_preview(product, changed, revised, candidate)
                check_draft(proposed_foreground, changed, revised, candidate)
            except ValueError as error:
                review_input.update(rejected_design=changed, layout_error=str(error))
                continue
            reason = required_text(review, "reason", "촬영 전 구도 검토")
            if original_valid:
                compare_task = BLOCKING_COMPARE
                compare_inputs = {"proposed_change": reason, "copy": inputs["copy"]}
                if format_issue:
                    compare_task += (" Return choice neither if both drafts repeat format_issue or only change "
                                     "position without visibly solving the reading path.")
                    compare_inputs["format_issue"] = format_issue
                    if has_failed_resize:
                        compare_task += " The last image is the failed resize. Use it to verify what visibly changed."
                comparison = ask_stage(ai, "ad_composition_compare", compare_task, compare_inputs,
                                       [to_float(first), to_float(rendered), *images[1:]])
                choice = comparison.get("choice")
                if choice == "neither" and format_issue:
                    review_input["layout_error"] = required_text(comparison, "reason", "게시 규격 구도 비교")
                    review_input["rejected_design"] = changed
                    photo_input = [to_float(first), to_float(rendered), *images]
                    task += "\nThe second image shows a rejected placement. Address layout_error without repeating it."
                    continue
                if choice == "original":
                    kept_reason = required_text(comparison, "reason", "촬영 전 구도 비교")
                    break
                if choice != "revision":
                    raise ValueError("촬영할 구도 시안을 고르지 못했습니다")
                reason = required_text(comparison, "reason", "촬영 전 구도 비교")
            first.save(folder / "before-composition.png")
            rendered.save(folder / "composition.png")
            for path in candidate.iterdir():
                path.replace(folder / path.name)
            if "graphic" in layout and "graphic" not in revised:
                (folder / layout["graphic"]["file"]).unlink()
            return changed, revised, reason
    if format_issue and not kept_reason:
        anchored = None
        if design["crop"] != "preserve" or not touches_photo_edge(foreground):
            anchored = measured_format_composition(ai, design, layout, folder, foreground, first, review_input)
        if not anchored:
            raise ValueError(f"게시 규격에 맞는 촬영 전 구도를 찾지 못했습니다. {review_input.get('layout_error', format_issue)}")
        changed, kept_reason = anchored
        anchored_layout = design_layers(changed, inputs["copy"], tuple(layout["size"]), inputs["fonts"], folder)
        rendered, foreground = composition_preview(product, changed, anchored_layout, folder)
        check_draft(foreground, changed, anchored_layout, folder)
        if graphic := layout.get("graphic"):
            (folder / graphic["file"]).unlink()
        first.save(folder / "before-composition.png")
        rendered.save(folder / "composition.png")
        return changed, anchored_layout, kept_reason
    first.save(folder / "composition.png")
    return design, layout, kept_reason


def find_foreground(scene, mask_model, threshold):
    from ..photo_retouch.product_mask import product_mask

    foreground = product_mask(mask_model, to_float(scene)) > threshold
    if not foreground.any():
        raise ValueError("생성한 사진에서 제품을 찾지 못했습니다")
    return foreground


PHOTO_REGIONS = """Locate the visible forms in this exact photograph that have to stay clear of typography
in addition to the product itself. For a physical action, follow the contours of hands, the receiver and the
material between them that make the action readable. Include the visible fingers or other parts even at the edge.
For a display where a distinct container holds the product, trace the container's visible outer shape, including
the lip or rim. The container and what it holds should read as one photographic form. Include this even in a
still life. Trace a separate recognizable object only when photo_cue explicitly names it as evidence for the
campaign, including an object required for a featured pairing. Leave out merely decorative props, tables,
pedestals, cast shadows, and backdrops. Follow each part's actual visible outline, leaving out the background
between parts and any implied motion path. Return JSON
with regions as an array of objects {part, points}. Use a short Korean name for part and at least 3 [x, y] vertices,
each at 0–100 percent from the left and top of the photograph. Return {"regions": []} when there are no
qualifying forms outside the product itself."""


def find_photo_regions(ai, photo, cue):
    answer = ask_stage(ai, "ad_photo_regions", PHOTO_REGIONS, {"photo_cue": cue}, [to_float(photo)])
    regions = answer.get("regions")
    if not isinstance(regions, list) or len(regions) > 8:
        raise ValueError("글자가 가리면 안 되는 촬영 영역을 확인하지 못했습니다")
    for region in regions:
        points = region.get("points") if isinstance(region, dict) else None
        valid_points = (isinstance(points, list) and len(points) >= 3
                        and all(isinstance(point, list) and len(point) == 2
                                and all(type(value) in (int, float) and 0 <= value <= 100 for value in point)
                                for point in points))
        if not valid_points or abs(sum(x * v - y * u for (x, y), (u, v) in
                                      zip(points, points[1:] + points[:1], strict=True))) < 1e-9:
            raise ValueError("보호 영역의 사진 좌표가 올바르지 않습니다")
        required_text(region, "part", "촬영 영역")
    return regions


def protected_photo_regions(layout):
    return (*layout.get("action_regions", ()), *layout.get("photo_regions", ()))


def photo_region_mask(regions, size):
    mask = Image.new("L", size)
    if regions:
        draw = ImageDraw.Draw(mask)
        for region in regions:
            draw.polygon([(round(x * (size[0] - 1) / 100), round(y * (size[1] - 1) / 100))
                          for x, y in region["points"]], fill=255)
        mask = mask.filter(ImageFilter.MaxFilter(2 * max(1, round(min(size) / 100)) + 1))
    return np.asarray(mask) > 128


def check_text_overlap(foreground, layout, folder):
    mask = np.zeros(tuple(reversed(layout["size"])), dtype=bool)
    headline_behind_subject = False
    x, y = layout["artwork"]["x"], layout["artwork"]["y"]
    height, width = foreground.shape
    mask[y:y + height, x:x + width] = foreground
    protected = None
    if regions := protected_photo_regions(layout):
        protected = np.zeros_like(mask)
        protected[y:y + height, x:x + width] = photo_region_mask(regions, (width, height))
    layers = [(f"{line['role']} 글자", line) for text in layout["texts"] for line in text.get("lines", (text,))]
    if graphic := layout.get("graphic"):
        layers.append(("색면", graphic))
    for name, layer_info in layers:
        with Image.open(folder / layer_info["file"]) as layer:
            ink = np.asarray(layer.getchannel("A")) > 128
        x, y = layer_info["x"], layer_info["y"]
        area = np.s_[y:y + layer_info["h"], x:x + layer_info["w"]]
        if protected is not None and (protected[area] & ink).any():
            raise ValueError(f"{name} 레이어가 보호할 사용 장면이나 용기 윤곽을 가립니다")
        covered = mask[area] & ink
        if layer_info.get("depth") == "behind_subject":
            if layer_info is graphic:
                if not covered.any() or not (ink & ~covered).any():
                    raise ValueError("제품 뒤 색면은 제품과 맞닿고 바깥에서도 보여야 합니다")
                continue
            if covered.any():
                from scipy.ndimage import label

                parts, _ = label(ink, structure=np.ones((3, 3), dtype=bool))
                sizes = np.bincount(parts[ink])
                hidden = np.bincount(parts[covered], minlength=len(sizes))
                if covered.sum() > 0.12 * ink.sum() or np.any((sizes > 2) & (hidden > 0.35 * sizes)):
                    raise ValueError("제품 뒤로 넘어가는 제목이 너무 많아 읽기 어렵습니다")
                headline_behind_subject = True
        elif covered.any():
            raise ValueError(f"{name} 레이어가 실제 제품을 가립니다")
    return headline_behind_subject


def tighten_aligned_stack(lines, foreground, layout, folder, *, shared_title=False):
    if (len(lines) != 2 or any(line.get("depth") == "behind_subject" for line in lines)
            or lines[0]["y"] + lines[0]["h"] > lines[1]["y"]):
        return
    paths = [folder / line["file"] for line in lines]
    original = []
    for path in paths:
        with Image.open(path) as layer:
            original.append(layer.convert("RGBA"))
    first, second = (layer.getchannel("A").getbbox() for layer in original)
    if first is None or second is None:
        return
    left, right = [(line["x"] + box[0], line["x"] + box[2])
                   for line, box in zip(lines, (first, second), strict=True)]
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap * 2 < min(left[1] - left[0], right[1] - right[0]):
        return
    desired = max(1, round(0.4 * min(first[3] - first[1], second[3] - second[1])))
    gap = lines[1]["y"] + second[1] - lines[0]["y"] - first[3]
    excess = max(0, gap - desired)
    # Use the shared title area while keeping the line facing the product.
    down = min(excess, original[0].height - first[3])
    up = min(excess - down, second[1])
    shifts = (0, 0)
    if shared_title and excess:
        product_rows = np.flatnonzero(foreground.any(axis=1))
        if product_rows.size and product_rows[-1] + layout["artwork"]["y"] < lines[0]["y"] + first[1]:
            down, up = 0, min(excess, second[1])
            shifts = (0, up - excess)
        else:
            up = 0
            shifts = (excess - down, 0)
    if not (down or up or any(shifts)):
        return
    for line, shift in zip(lines, shifts, strict=True):
        line["y"] += shift
    for path, layer, box, shift in zip(paths, original, (first, second), (down, -up), strict=True):
        if not shift:
            continue
        shifted = Image.new("RGBA", layer.size)
        shifted.paste(layer.crop(box), (box[0], box[1] + shift))
        shifted.save(path)
    try:
        check_text_overlap(foreground, layout, folder)
    except ValueError:
        for line, shift in zip(lines, shifts, strict=True):
            line["y"] -= shift
        for path, layer in zip(paths, original, strict=True):
            layer.save(path)


def tighten_text_groups(foreground, layout, folder, changed=None):
    roles = {text["role"]: text for text in layout["texts"]}
    if changed is None or "headline" in changed:
        tighten_aligned_stack(roles["headline"].get("lines", ()), foreground, layout, folder, shared_title=True)
    if changed is None or {"product_name", "price"} <= changed.keys():
        offer = tuple(sorted((roles["product_name"], roles["price"]), key=lambda text: text["y"]))
        tighten_aligned_stack(offer, foreground, layout, folder)


def rendered_text_bounds(layout, folder):
    size = tuple(layout["size"])
    bounds_by_role = {}
    for text in layout["texts"]:
        line_bounds = []
        for line in text.get("lines", (text,)):
            with Image.open(folder / line["file"]) as image:
                bounds = image.getchannel("A").getbbox()
            if bounds:
                ink = [round(100 * (edge + offset) / side)
                       for edge, offset, side in zip(bounds, (line["x"], line["y"]) * 2, size * 2, strict=True)]
                bounds_by_role[line["role"]] = ink
                line_bounds.append(ink)
        if "lines" in text and line_bounds:
            bounds_by_role[text["role"]] = [min(b[0] for b in line_bounds), min(b[1] for b in line_bounds),
                                             max(b[2] for b in line_bounds), max(b[3] for b in line_bounds)]
    return bounds_by_role


def weaker_headline_focus(layout, ink, phrase):
    headline = next(text for text in layout["texts"] if text["role"] == "headline")
    lines = headline.get("lines")
    if not lines:
        return False
    matching = [line["role"] for line in lines if f" {phrase} " in f" {line['text']} "]
    if len(matching) != 1 or matching[0] not in ink:
        return True
    primary = ink[matching[0]][3] - ink[matching[0]][1]
    return any(ink[role][3] - ink[role][1] > primary * 1.15
               for role in (line["role"] for line in lines) if role != matching[0] and role in ink)


def find_headline_priority(ai, layout, ink, brief, scene):
    headline = next(text for text in layout["texts"] if text["role"] == "headline")
    lines = headline.get("lines")
    if not brief or not brief.get("message") or not lines:
        return None
    if not any(weaker_headline_focus(layout, ink, line["text"]) for line in lines):
        return None
    reading_order = [line["text"] for line in lines]
    product = next((text["text"] for text in layout["texts"] if text["role"] == "product_name"), None)
    decision = ask_stage(ai, "ad_headline_priority", HEADLINE_PRIORITY,
                         {"message": brief["message"], "product_name": product, "headline_lines": reading_order},
                         [to_float(scene)])
    phrase = decision.get("phrase")
    reason = decision.get("reason")
    if not isinstance(phrase, str) or not phrase.strip() or not isinstance(reason, str) or not reason.strip():
        return None
    if sum(f" {phrase} " in f" {line} " for line in reading_order) != 1:
        return None
    if weaker_headline_focus(layout, ink, phrase):
        return {"phrase": phrase, "reason": reason}
    return None


def transfer_headline_scale(design, layout, phrase):
    headline = design["styles"]["headline"]
    title = next(text for text in layout["texts"] if text["role"] == "headline")
    if len(headline.get("lines") or ()) != 2:
        return None
    first, second = headline["lines"]
    sizes = [line.get("size", headline["size"]) for line in (first, second)]
    index = next(i for i, line in enumerate(title["lines"]) if f" {phrase} " in f" {line['text']} ")
    if sizes[index] >= sizes[1 - index]:
        return None
    x0, y0, w0, h0 = first["box"]
    x1, y1, w1, h1 = second["box"]
    lines = [first | {"size": sizes[1], "box": [x0, y0, w0, h1]},
             second | {"size": sizes[0], "box": [x1, y1 + h1 - h0, w1, h0]}]
    return design["styles"] | {"headline": headline | {"lines": lines}}


def check_offer_spacing(ink, size):
    if not {"product_name", "price"} <= ink.keys():
        raise ValueError("상품명과 가격이 모두 화면에 보여야 합니다")
    first, second = sorted((ink["product_name"], ink["price"]))
    if min(first[3], second[3]) <= max(first[1], second[1]):
        return
    width, height = size
    gap = (second[0] - first[2]) * width
    letter_height = max(first[3] - first[1], second[3] - second[1]) * height
    if gap > 3 * letter_height:
        raise ValueError("상품명과 가격 사이의 빈 공간이 너무 넓습니다. 제목과 사진은 두고 두 문구를 가까이 배치하세요")


def check_type_margin(size, ink):
    width, height = size
    clearance = min(size) / 40
    for role, (left, top, right, bottom) in ink.items():
        edges = {"왼쪽": left * width, "위쪽": top * height,
                 "오른쪽": (100 - right) * width, "아래쪽": (100 - bottom) * height}
        side = min(edges, key=edges.get)
        if edges[side] < 100 * clearance:
            shift = ceil(clearance - edges[side] / 100)
            raise ValueError(f"{role} 글자가 포스터 {side} 가장자리에 너무 가깝습니다. "
                             f"글자 묶음을 안쪽으로 {shift}픽셀 이상 옮겨 주세요")


def align_nearby_offer(plan, ink, size):
    roles = ("product_name", "price")
    if plan.get("accent") or not {"headline", *roles} <= ink.keys():
        return None
    alignment = plan["styles"]["headline"]["align"]
    if any(plan["styles"][role]["align"] != alignment for role in roles):
        return None
    heading = ink["headline"]
    offer = [ink[role] for role in roles]
    if not (heading[3] <= min(box[1] for box in offer) or max(box[3] for box in offer) <= heading[1]):
        return None

    def anchor(left, right):
        return {"left": left, "center": (left + right) / 2, "right": right}[alignment]

    start, end = min(box[0] for box in offer), max(box[2] for box in offer)
    dx = round(anchor(heading[0], heading[2]) - anchor(start, end))
    if not dx or abs(dx) * size[0] > 5 * min(size):
        return None
    boxes = plan["boxes"].copy()
    for role in roles:
        x, y, w, h = boxes[role]
        boxes[role] = [x + dx, y, w, h]
    return plan | {"boxes": boxes}


def expanded_outer_margins(before, after):
    # The threshold triggers a visual comparison. The margin can still be useful.
    return {edge: space - before[edge] for edge, space in after.items() if space - before[edge] >= 5}


def balanced_text_positions(original, revision, size, before, after):
    axis, ends = (1, ("top", "bottom")) if size[1] >= size[0] else (0, ("left", "right"))
    expanded = expanded_outer_margins(before, after)
    if expanded:
        edge_axes = {"left": 0, "right": 0, "top": 1, "bottom": 1}
        edge = max(expanded, key=lambda side: expanded[side] * size[edge_axes[side]])
        axis = edge_axes[edge]
    elif any(after[edge] <= before[edge] for edge in ends):
        return None
    movement = {role: original["boxes"][role][axis] - box[axis] for role, box in revision["boxes"].items()}
    offsets = {role: round(amount / 2) for role, amount in movement.items()}

    def shift(box, amount):
        moved = box.copy()
        moved[axis] += amount
        return moved

    styles = revision["styles"].copy()
    headline = styles["headline"]
    if "lines" in headline:
        styles["headline"] = headline | {
            "lines": [line | {"box": shift(line["box"], offsets["headline"])} for line in headline["lines"]]}
    balanced = revision | {"styles": styles,
                           "boxes": {role: shift(box, offsets[role]) for role, box in revision["boxes"].items()}}
    accent, original_accent = revision.get("accent"), original.get("accent")
    if accent and original_accent and accent["shape"] == original_accent["shape"]:
        amount = original_accent["box"][axis] - accent["box"][axis]
        if amount and amount == movement.get("product_name") == movement.get("price"):
            balanced["accent"] = accent | {"box": shift(accent["box"], round(amount / 2))}
    return balanced


def review_finished(ai, answer, layout, scene, foreground, folder, design_input, references, *,
                    assess_composition=False, keep_wrap=False):
    """완성 사진 위의 글자와 구도를 검사하고 사진을 유지한 채 배치를 고칩니다."""
    from .format_layout import outer_margins

    if "photo_regions" not in layout:
        layout["photo_regions"] = find_photo_regions(ai, scene, layout.get("photo_cue"))
    rebuilt_format = (design_input.get("revision_feedback") or {}).get("format_issue")

    def check_layout(candidate, directory):
        check_text_overlap(foreground, candidate, directory)
        tighten_text_groups(foreground, candidate, directory)
        if protected_photo_regions(candidate) or rebuilt_format:
            ink = rendered_text_bounds(candidate, directory)
            check_type_margin(candidate["size"], ink)
            if rebuilt_format:
                check_offer_spacing(ink, candidate["size"])

    original_error, contrast_issue = None, None
    try:
        check_layout(layout, folder)
    except ValueError as error:
        original_error = str(error)
    else:
        answer, layout, contrast_issue = readable_text_colors(answer, layout, folder, foreground)
    if original_error is None and contrast_issue is None and not assess_composition:
        return answer, layout, None
    first = compose(layout, folder)
    artwork = layout["artwork"]
    size = tuple(layout["size"])
    rows, columns = np.nonzero(foreground)
    placement = [round(100 * value / limit) for value, limit in zip(
        (artwork["x"] + columns.min(), artwork["y"] + rows.min(), artwork["x"] + columns.max() + 1,
         artwork["y"] + rows.max() + 1), (*size, *size), strict=True)]
    inputs = design_input | {"design": answer, "product_bounds_left_top_right_bottom": placement,
                             "headline_breaks": headline_breaks(design_input["copy"]["headline"]),
                             "layout_error": original_error, "rendered_text_bounds": rendered_text_bounds(layout, folder)}
    inputs["original_outer_space"] = outer_margins(placement, inputs["rendered_text_bounds"].values())
    task = FINAL_REVIEW
    if regions := protected_photo_regions(layout):
        inputs["photo_regions"] = regions
        task += ("\nThe partly tinted photograph after the plain photograph marks the visible action and container "
                 "forms that must stay clear. Keep letter strokes and graphics off the tinted area. The product "
                 "name and price should share a nearby alignment in calm space. Leave at least 2.5% of the "
                 "short canvas side around actual ink. When one side is too narrow, use open space on the other "
                 "side before squeezing the offer against the edge. Avoid changing unrelated text.")
        tinted = Image.blend(scene.convert("RGB"), Image.new("RGB", scene.size, "#ff5c7a"), 0.65)
        guide = Image.composite(tinted, scene.convert("RGB"), Image.fromarray(photo_region_mask(regions, scene.size)))
        layout_references = [to_float(guide), *references]
    else:
        layout_references = references
    requested_alternative = "composition_issue" in inputs
    priority = None
    if assess_composition and not requested_alternative:
        priority = find_headline_priority(ai, layout, inputs["rendered_text_bounds"],
                                          design_input.get("brief") or layout.get("brief"), scene)
    if priority:
        inputs["headline_priority"] = priority
        inputs["composition_issue"] = (f"광고의 핵심인 ‘{priority['phrase']}’가 제목의 다른 줄보다 작게 보입니다. "
                                       "사진과 판매 정보는 유지하고 제목의 시각적 비중을 바로잡으세요.")
        task += ("\nUse headline_priority to adjust the relative size or weight of the headline lines. "
                 "Transfer the current strongest line's visual scale to the focus words where there is room. "
                 "Use rendered_text_bounds to keep the title group's connection to the photograph. "
                 "A smaller secondary line alone can leave the whole headline too weak or too far from the product. "
                 "Use the same overall title space with new line boxes when feasible. Keep the exact wording, "
                 "natural reading order, and positions of the photo and supporting information.")
    if contrast_issue:
        inputs.setdefault("composition_issue", contrast_issue)
        inputs["contrast_issue"] = contrast_issue
        task += ("\nUse contrast_issue to move the affected words onto an area of consistent brightness. "
                 "Keep the photograph and alignments that are already working. Move the product name and price "
                 "together when both are used as one group.")
    needs_alternative = "composition_issue" in inputs
    if needs_alternative:
        task += ("\nAddress composition_issue on the actual poster. Design the reading path as a whole: place the "
                 "headline by the photographic feature, then align and group the name and price so they do not "
                 "form an unrelated block. Blend an arbitrary seam that divides the reading path. If a panel "
                 "is appropriate, use its edge to align the elements. Give the headline enough scale for the phone feed. "
                 "Check the whole canvas after moving type, including the larger outer margins a move can leave. "
                 "Return an actual alternative with at least one changed field. Both rendered posters will be compared later.")

    def render_revision(plan, folder):
        updated = design_layers(plan, design_input["copy"], size, design_input["fonts"], folder)
        updated.update({key: layout[key] for key in ("action_regions", "photo_regions") if key in layout})
        revised_scene = (scene if plan["edge"] == answer["edge"] else
                         set_photo_edge(scene, foreground, artwork, size, plan["edge"]))
        revised_scene.save(folder / artwork["file"])
        Image.fromarray(foreground).save(folder / "foreground.png")
        check_layout(updated, folder)
        plan, updated, remaining = readable_text_colors(plan, updated, folder, foreground)
        if remaining:
            raise ValueError(remaining)
        return plan, updated

    typography_only = not requested_alternative and (contrast_issue or priority)
    editable = ("boxes", "styles") if typography_only else ("boxes", "styles", "accent", "edge")
    transferred = transfer_headline_scale(answer, layout, priority["phrase"]) if priority else None
    for attempt in range(2):
        if attempt == 0 and transferred is not None:
            review = {"revision": {"styles": transferred}}
        else:
            review = ask_stage(ai, "ad_final_layout", task, inputs,
                               [to_float(first), to_float(scene), *layout_references])
        revision = review.get("revision")
        if "revision" not in review:
            revision = {key: review[key] for key in editable if key in review} or None
        if not isinstance(revision, dict):
            if original_error is None and not needs_alternative:
                break
            inputs["layout_error"] = "제출한 배치 수정이 빠졌습니다."
            if original_error:
                inputs["layout_error"] = f"{original_error}. {inputs['layout_error']}"
            continue
        accent = revision.get("accent")
        styles = revision.get("styles", answer["styles"])
        if (isinstance(accent, dict) and accent.keys() == {"phrase", "color"}
                and isinstance(styles, dict) and isinstance(styles.get("headline"), dict)):
            revision = {key: value for key, value in revision.items() if key != "accent"}
            revision["styles"] = styles | {"headline": {"emphasis": accent} | styles["headline"]}
        revised = answer | {key: revision[key] for key in editable if key in revision}
        styles = revised["styles"]
        headline = styles.get("headline") if isinstance(styles, dict) else None
        if isinstance(headline, dict) and (emphasis := headline.get("emphasis")) is not None:
            if not valid_headline_emphasis(emphasis, design_input["copy"]["headline"]):
                headline = headline.copy()
                headline.pop("emphasis")
                original = answer["styles"]["headline"]
                if "emphasis" in original:
                    headline["emphasis"] = original["emphasis"]
                revised["styles"] = styles | {"headline": headline}
        if keep_wrap and isinstance(revised["styles"], dict):
            styles = {}
            for role, style in revised["styles"].items():
                original = answer["styles"].get(role)
                if isinstance(style, dict) and original:
                    style = {key: value for key, value in style.items() if key not in ("max_lines", "break_after")}
                    style.update({key: original[key] for key in ("max_lines", "break_after") if key in original})
                    if "emphasis" in original:
                        style.setdefault("emphasis", original["emphasis"])
                styles[role] = style
            revised["styles"] = styles
        if revised == answer:
            if original_error is None and not needs_alternative:
                break
            inputs["layout_error"] = "이전과 같은 배치입니다."
            if original_error:
                inputs["layout_error"] = f"{original_error}. {inputs['layout_error']}"
            continue
        with TemporaryDirectory(dir=folder) as staging:
            candidate = Path(staging) / "revision"
            candidate.mkdir()
            proposed = revised
            try:
                revised, updated = render_revision(revised, candidate)
            except ValueError as error:
                inputs = inputs | {"rejected_design" if needs_alternative else "design": revised,
                                   "layout_error": str(error)}
                continue
            if regions and original_error is not None:
                aligned = align_nearby_offer(revised, rendered_text_bounds(updated, candidate), size)
                if aligned is not None:
                    aligned_dir = Path(staging) / "aligned"
                    aligned_dir.mkdir()
                    try:
                        aligned, aligned_layout = render_revision(aligned, aligned_dir)
                    except ValueError:
                        pass
                    else:
                        revised, updated, candidate = aligned, aligned_layout, aligned_dir
            revision_ink = rendered_text_bounds(updated, candidate)
            revision_outer_space = outer_margins(placement, revision_ink.values())
            expanded = expanded_outer_margins(inputs["original_outer_space"], revision_outer_space)
            if original_error is None and (needs_alternative or expanded):
                if priority and weaker_headline_focus(updated, revision_ink, priority["phrase"]):
                    inputs = inputs | {"rejected_design": revised,
                                       "layout_error": "수정한 제목에서도 광고의 핵심 표현이 더 작게 보입니다."}
                    continue
                evidence = {
                    "composition_issue": (inputs.get("composition_issue") or review.get("reason")
                                          or "글자의 새 배치와 전체 여백을 비교합니다."),
                    "copy": inputs["copy"],
                    "product_bounds": placement,
                    "original_ink": inputs["rendered_text_bounds"],
                    "revision_ink": revision_ink,
                    "original_outer_space": inputs["original_outer_space"],
                    "revision_outer_space": revision_outer_space,
                    "expanded_outer_space": expanded,
                }
                if priority:
                    evidence["headline_priority"] = priority
                versions = {"revision": (revised, updated, candidate)}
                pictures = [to_float(first), to_float(compose(updated, candidate))]
                balanced = balanced_text_positions(answer, proposed, size, evidence["original_outer_space"],
                                                   evidence["revision_outer_space"])
                if balanced is not None and balanced != proposed:
                    alternate = Path(staging) / "balanced"
                    alternate.mkdir()
                    try:
                        balanced, balanced_layout = render_revision(balanced, alternate)
                    except ValueError:
                        pass
                    else:
                        balanced_ink = rendered_text_bounds(balanced_layout, alternate)
                        if not priority or not weaker_headline_focus(balanced_layout, balanced_ink, priority["phrase"]):
                            evidence.update(balanced_ink=balanced_ink,
                                            balanced_outer_space=outer_margins(placement, balanced_ink.values()))
                            versions["balanced"] = (balanced, balanced_layout, alternate)
                            pictures.append(to_float(compose(balanced_layout, alternate)))
                comparison = ask_stage(ai, "ad_final_composition_compare", FINISHED_COMPOSITION_COMPARE, evidence,
                                       [*pictures, *references])
                reason = required_text(comparison, "reason", "완성 배치 비교")
                choice = comparison.get("choice")
                if choice == "original":
                    inputs = inputs | {"rejected_design": revised, "layout_error": reason,
                                       "expanded_outer_space": expanded}
                    continue
                if not isinstance(choice, str) or choice not in versions:
                    raise ValueError("어느 배치가 더 나은지 확인하지 못했습니다")
                revised, updated, candidate = versions[choice]
                review["reason"] = reason
            first.save(folder / "before-typesetting.png")
            for path in candidate.iterdir():
                path.replace(folder / path.name)
            if "graphic" in layout and "graphic" not in updated:
                (folder / layout["graphic"]["file"]).unlink()
            return revised, updated, review.get("reason")
    if original_error is not None:
        raise ValueError(original_error)
    if requested_alternative:
        raise ValueError(inputs["layout_error"])
    return answer, layout, None


def review_crowded_offer(ai, design, layout, foreground, folder, fonts):
    from scipy.ndimage import binary_dilation
    from .format_layout import offer_reflow_options

    ink = rendered_text_bounds(layout, folder)
    if design.get("accent") or "price" not in ink:
        return design, layout, None
    size = tuple(layout["size"])
    width, height = size
    _, top, _, bottom = ink["price"]
    short_side = min(size)
    if (bottom - top) * height > 3 * short_side or (100 - bottom) * height > 4 * short_side:
        return design, layout, None

    artwork = layout["artwork"]
    ax, ay, aw, ah = (artwork[key] for key in ("x", "y", "w", "h"))
    mask = np.zeros((height, width), dtype=bool)
    mask[ay:ay + ah, ax:ax + aw] = foreground | photo_region_mask(protected_photo_regions(layout), (aw, ah))
    occupied = binary_dilation(mask, iterations=max(1, round(short_side / 100)))
    rows, columns = np.nonzero(occupied)
    bounds = [round(100 * value / length) for value, length in zip(
        (columns.min(), rows.min(), columns.max() + 1, rows.max() + 1), size * 2, strict=True)]
    options = offer_reflow_options(layout, design, occupied, bounds)
    if not options:
        return design, layout, None

    copy = {text["role"]: text["text"] for text in layout["texts"]}
    first = compose(layout, folder)
    with Image.open(folder / artwork["file"]) as image, TemporaryDirectory(dir=folder) as temporary:
        images, candidates = [to_float(first)], {}
        for name, plan in options.items():
            directory = Path(temporary) / name
            directory.mkdir()
            try:
                revised = design_layers(plan, copy, size, fonts, directory)
                revised.update({key: layout[key] for key in ("action_regions", "photo_regions") if key in layout})
                image.save(directory / artwork["file"])
                Image.fromarray(foreground).save(directory / "foreground.png")
                check_text_overlap(foreground, revised, directory)
                updated_ink = rendered_text_bounds(revised, directory)
                check_type_margin(size, updated_ink)
                check_offer_spacing(updated_ink, size)
                price = updated_ink["price"]
                if price[3] - price[1] <= bottom - top:
                    continue
                plan, revised, contrast_issue = readable_text_colors(plan, revised, directory, foreground)
                if contrast_issue:
                    continue
            except ValueError:
                continue
            images.append(to_float(compose(revised, directory)))
            candidates[name] = (plan, revised, directory)
        if not candidates:
            return design, layout, None
        comparison = ask_stage(ai, "ad_crowded_offer_compare", CROWDED_OFFER_COMPARE,
                               {"option_order": ["original", *candidates], "copy": copy}, images)
        choice, reason = comparison.get("choice"), comparison.get("reason")
        if not isinstance(choice, str) or choice not in candidates or not isinstance(reason, str) or not reason.strip():
            return design, layout, None
        design, revised, directory = candidates[choice]
        first.save(folder / "before-offer-reflow.png")
        for path in directory.iterdir():
            if path.name not in (artwork["file"], "foreground.png"):
                path.replace(folder / path.name)
        return design, layout | revised, reason


def blend_edges(scene, foreground=None, *, position=(0, 0), canvas_size=None):
    feather = max(1, round(min(scene.size) * 0.16))
    x, y = np.arange(scene.width), np.arange(scene.height)
    left, top = position
    right, bottom = (left + scene.width, top + scene.height)
    # 캔버스 끝에는 이웃하는 배경이 없으므로 사진을 그대로 둡니다.
    horizontal = np.minimum(x if canvas_size is None or left > 0 else np.full_like(x, feather),
                            scene.width - 1 - x if canvas_size is None or right < canvas_size[0]
                            else np.full_like(x, feather))
    vertical = np.minimum(y if canvas_size is None or top > 0 else np.full_like(y, feather),
                          scene.height - 1 - y if canvas_size is None or bottom < canvas_size[1]
                          else np.full_like(y, feather))
    distance = np.minimum(horizontal[None, :], vertical[:, None])
    matte = (np.clip(distance / feather, 0, 1) * 255).astype(np.uint8)
    if foreground is not None:
        matte[foreground] = 255
    scene = scene.convert("RGBA")
    scene.putalpha(Image.fromarray(matte))
    return scene


def set_photo_edge(scene, foreground, artwork, size, edge):
    revised = (blend_edges(scene, position=(artwork["x"], artwork["y"]), canvas_size=size)
               if edge == "blend" else scene.convert("RGB").convert("RGBA"))
    alpha = revised.getchannel("A")
    alpha.paste(scene.convert("RGBA").getchannel("A"), mask=Image.fromarray(foreground))
    revised.putalpha(alpha)
    return revised


def preview_artwork(photo, artwork, canvas_size, design):
    picture_size = (artwork["w"], artwork["h"])
    x, y = artwork["x"], artwork["y"]
    if design["crop"] == "intentional":
        picture = ImageOps.fit(photo, picture_size, Image.Resampling.LANCZOS)
        return (blend_edges(picture, position=(x, y), canvas_size=canvas_size)
                if design["edge"] == "blend" else picture)
    if design["edge"] == "framed":
        return ImageOps.pad(photo, picture_size, Image.Resampling.LANCZOS, color=design["background"])
    picture = ImageOps.contain(photo, picture_size, Image.Resampling.LANCZOS)
    offset_x, offset_y = (artwork["w"] - picture.width) // 2, (artwork["h"] - picture.height) // 2
    picture = blend_edges(picture, position=(x + offset_x, y + offset_y), canvas_size=canvas_size)
    area = Image.new("RGBA", picture_size)
    area.alpha_composite(picture, (offset_x, offset_y))
    return area


def make_scene_ad(photo, fields, idea, models, folder, *, references=(), preview=False, proposal=None, previous=None,
                  rejected=None, feedback=None, scene_issue=None, start_direction=None, viewpoint=None,
                  campaign_intent=None, format_rebuild=None):
    from ..photo_retouch.product_mask import product_mask

    if folder.exists():
        raise FileExistsError(folder)
    settings = load_settings()
    ai = models["ai"]
    if proposal and previous:
        raise ValueError("저장한 시안 적용과 완성본 개선 중 하나를 선택하세요")
    if rejected and not previous:
        raise ValueError("보류한 후보를 전달하려면 비교 기준이 되는 완성본도 필요합니다")
    if feedback and not previous:
        raise ValueError("장면을 수정하려면 기준이 되는 완성본이 필요합니다")
    if campaign_intent and (not previous or feedback):
        raise ValueError("새 연출에는 기존 포스터와 하나의 연출 의도가 필요합니다")
    if format_rebuild is not None and (not previous or feedback or campaign_intent or rejected or viewpoint):
        raise ValueError("게시 규격에 맞춘 새 구도에는 채택한 포스터가 필요합니다")
    prior = load_layout(previous) if previous else None
    photo_cue = required_text(proposal, "photo_cue", "촬영 연출") if proposal and "photo_cue" in proposal else None
    photo_checks = []
    kept = prior or proposal
    if kept and kept["fields"] != fields:
        raise ValueError("시안의 상품 정보가 현재 입력과 다릅니다")
    brief = kept["brief"] if kept else plan_copy(ai, photo, fields, idea, start_direction=start_direction)
    copy = {"headline": brief["headline"], "product_name": fields["product_name"],
            "price": f"{fields['price_krw']:,}원"}
    canvas = settings["canvas"]
    size = (canvas["width"], canvas["height"])
    if format_rebuild is not None:
        size, format_issue, failed_resize = format_rebuild
        copy = {text["role"]: text["text"] for text in prior["texts"]}
    images = [to_float(photo), *map(to_float, references)]
    revision_info = {"requested_scene_change": feedback} if feedback else None
    design_images = images
    mask_model = models.get("mask")
    threshold = settings["product"]["mask_threshold"]
    product, source_mask = None, None
    parts = None
    staging_intent = None
    if prior:
        if prior.get("preview"):
            raise ValueError("사진 생성이 끝난 광고만 개선에 사용할 수 있습니다")
        with Image.open(previous / "ad.png") as finished:
            prior_image = to_float(finished)
        if feedback or format_rebuild is not None:
            design_images = [images[0], prior_image, *images[1:]]
    if format_rebuild is not None:
        revision_info = {"format_issue": format_issue}
    elif prior and not feedback:
        previous_design = prior.get("design", {})
        inputs = {"brief": brief, "copy": copy, "previous_concept": previous_design.get("concept", brief["message"])}
        inputs["previous_art_direction"] = previous_design.get("art_direction", prior.get("scene_prompt"))
        inputs["previous_visual_review"] = prior.get("revision", {}).get("visual_review")
        critique_prompt, critique_images = CRITIQUE, [prior_image, *images]
        fields_to_keep = ("weakness", "hypothesis")
        campaign_viewpoints = {}
        if campaign_intent:
            if mask_model is not None:
                product, source_mask = extract_subject(photo, mask_model, threshold)
                parts = source_parts(source_mask)
            inputs["editor_intent"] = campaign_intent
            critique_prompt += ("\nDesign one replacement campaign from editor_intent. The full composition may change. "
                                "Do not salvage the old subject placement or type merely for continuity. In hypothesis, "
                                "describe a physically visible scene with the actual product. Also return layout_device "
                                "(how positioning, type or a graphic makes that same idea visible) and visible_result "
                                "(what must be apparent in the rendered poster at phone size). Use a layout device "
                                "the editor can draw: straight headline lines, size, selective color, a rectangular or "
                                "elliptical shape, or a small overlap behind the product. Do not plan curved lettering.")
            fields_to_keep += ("layout_device", "visible_result")
            campaign_viewpoints = available_viewpoints()
            if campaign_viewpoints:
                inputs["camera_viewpoints"] = campaign_viewpoints
                critique_prompt += (" If the concept needs a visibly different camera position relative to the source, "
                                    "return viewpoint as one key from camera_viewpoints. This will move the actual camera "
                                    "reference. Return null when the scene does not need a new camera position.")
            if parts is not None:
                critique_prompt += (" If moving one clearly separable piece already visible in the original photo is "
                                    "essential to the hypothesis, return staging_intent describing what moves, where, "
                                    "and why. That piece will actually be moved in the reference without changing its size "
                                    "or count. Use this only where the original exposes the whole visible silhouette. "
                                    "Otherwise return null. Use at most one of viewpoint and staging_intent.")
        if rejected:
            discarded = load_layout(rejected)
            if discarded["fields"] != fields:
                raise ValueError("보류한 후보의 상품 정보가 현재 입력과 다릅니다")
            if discarded.get("preview"):
                raise ValueError("사진 생성이 끝난 광고만 보류한 후보로 사용할 수 있습니다")
            inputs["rejected_attempt"] = {"art_direction": discarded["scene_prompt"],
                                          "visual_review": discarded.get("revision", {}).get("visual_review")}
            with Image.open(rejected / "ad.png") as rejected_image:
                critique_images.append(to_float(rejected_image))
            critique_prompt += "\nThe last image is a rejected attempt. Address its weakness and avoid repeating it."
        criticism = ask_stage(ai, "ad_cycle_critique", critique_prompt, inputs, critique_images)
        revision_info = {key: required_text(criticism, key, "완성본 검토") for key in fields_to_keep}
        if parts is not None and criticism.get("staging_intent") is not None:
            staging_intent = required_text(criticism, "staging_intent", "완성본 검토")
            revision_info["staging_intent"] = staging_intent
        elif campaign_viewpoints and (camera_choice := criticism.get("viewpoint")) is not None:
            if not isinstance(camera_choice, str) or camera_choice not in campaign_viewpoints:
                raise ValueError("새 연출에 사용할 촬영 시점을 확인하지 못했습니다")
            viewpoint = camera_choice
    design_input = {"brief": brief, "copy": copy, "fonts": list(settings["fonts"]), "canvas_size": size,
                    "headline_breaks": headline_breaks(copy["headline"])}
    if revision_info:
        design_input.update(brief={"message": brief["message"], "evidence": brief.get("evidence")},
                            idea=idea, revision_feedback=revision_info)
    art_plan = kept.get("art_direction_plan") if proposal or format_rebuild is not None else None
    if art_plan is None and not proposal:
        art_plan = plan_art_direction(ai, fields, design_input["brief"] | {"headline": copy["headline"]}, idea,
                                      images, revision_info)
    if mask_model is not None and source_mask is None:
        product, source_mask = extract_subject(photo, mask_model, threshold)
    group = None
    if format_rebuild is not None and product is not None and prior["design"].get("source_crop") is None:
        parts = source_parts(source_mask)
        if parts is not None:
            group = stage_group(ai, photo, parts, size, prior_image, format_issue, copy)
            if group is None:
                staging = ask_stage(ai, "ad_format_staging", FORMAT_STAGING,
                                    {"target_canvas": size, "format_issue": format_issue, "copy": copy},
                                    [images[0], prior_image, to_float(on_gray(product))])
                if staging.get("move") is not None:
                    staging_intent = required_text(staging, "move", "게시 규격의 상품 배치")
                    revision_info["staging_intent"] = staging_intent
    moved_view = None
    staged_view = None
    moved_piece = None
    if group is not None:
        product, moved_piece, observation = group
        revision_info.update(staged_group=True, staging_observation=observation)
        staged_view = to_float(on_gray(product))
    elif staging_intent:
        if product is None:
            raise ValueError("새 구도에 옮길 상품 영역을 찾지 못했습니다")
        product, moved_piece, observation = stage_subject(ai, photo, source_mask, parts, staging_intent)
        revision_info["staging_observation"] = observation
        staged_view = to_float(on_gray(product))
    elif viewpoint:
        if product is None:
            raise ValueError("새 시점에서 보여 줄 상품 영역을 찾지 못했습니다")
        moved_view = camera_view(photo, source_mask, viewpoint)
    source_framing = proposal.get("source_framing") if proposal else None
    if staged_view is not None:
        source_framing = {"source_crop": None, "reason": revision_info["staging_observation"]}
    elif moved_view is not None:
        source_framing = {"source_crop": None, "reason": "새 촬영 시점에서 상품의 구성을 유지합니다"}
    elif format_rebuild is not None:
        crop = prior["design"].get("source_crop")
        source_framing = {"source_crop": crop, "reason": "채택한 광고의 상품 촬영 범위를 유지합니다"}
    elif not proposal and product is not None:
        framing_input = design_input | {"art_direction_plan": art_plan, "idea": idea}
        if scene_issue:
            framing_input = {"brief": brief, "copy": copy, "idea": idea, "scene_issue": scene_issue,
                             "visible_feature": art_plan["evidence"]}
        framing_input["product_facts"] = fields
        source_framing = choose_source_framing(ai, product, framing_input, images)
        source_crop = source_framing["source_crop"]
        scale = "detail" if source_crop is not None else art_plan["shot_scale"]
        if source_crop is None and scale == "detail":
            scale = "close"
        art_plan = art_plan | {"shot_scale": scale}
    if source_framing:
        design_input["source_framing"] = source_framing
    if art_plan:
        design_input["art_direction_plan"] = art_plan | {"layout_guide": THEMES[art_plan["theme"]]["layout"]}

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as staging:
        result = Path(staging)
        design_task = (DESIGN + "\nUse the selected layout_guide and connect the chosen photo_device to one "
                       "clear reading path. Return a JSON design object.\n" + DESIGN_SCHEMA)
        frame_fields = {}
        if source_framing and not proposal:
            crop = source_framing["source_crop"]
            frame_fields = {"source_crop": crop, "crop": "preserve" if crop is None else "intentional"}
            design_task += "\nSource framing is approved. Use source_framing and lay out the actual view without changing it."
        if format_rebuild is not None:
            design_input["approved_concept"] = prior["design"]["concept"]
            design_input["has_failed_resize"] = failed_resize is not None
            design_task += ("\nThe second image is the selected campaign poster. Keep the existing campaign idea, "
                            "visible product assortment, visual motif and typographic voice. Solve format_issue by "
                            "planning the photograph and type together for canvas_size from the beginning. Think in the "
                            "long axis of the new frame. Use camera depth, the direction of the photographic cue or "
                            "relative subject and headline scale. Do not center the old composition between two "
                            "background-only bands or invent new product components. Treat the first image as identity.")
            if failed_resize is not None:
                design_task += (" The last image shows the failed resize. Use its visible product, scene detail and "
                                "copy positions to locate the problem. Reblock the product and defining scene element "
                                "relative to each other and the copy so the layout uses the new frame from edge to edge. "
                                "Leave purposeful quiet space around the copy.")
        elif feedback:
            design_task += ("\nThe second image is the previous poster. Plan the requested_scene_change in revision_feedback "
                            "and preserve unaffected parts of that composition.")
            if scene_issue:
                design_task += " The approved source_framing takes precedence over framing suggested by the editor."
            else:
                design_task += " Do not substitute another change."
        elif revision_info:
            design_task += ("\nMake room for the visual idea in revision_feedback. Give the photography "
                            "and type a new shared composition with the same exact copy.")
            if campaign_intent:
                design_task += (" Use layout_device and show visible_result. Draw the relationship in the actual layout "
                                "through relative scale, overlap or direction; a separate caption above a catalog shot is insufficient.")
        if staged_view is not None:
            design_images = [*design_images, staged_view]
            location = "The image before the failed resize" if design_input.get("has_failed_resize") else "The final image"
            design_task += (f"\n{location} is the actual rearranged product. Use its new positions and physical "
                            "shape when directing the layout. Gray is transparent.")
            if group is not None:
                design_task += (" Use the full rearranged silhouette as the shape of subject_box. Give its leading "
                                "piece space for the planned visual action and fit the typography into the adjacent "
                                "quiet part of the same scene. Do not flatten the photographed rows back into a line.")
        elif moved_view is not None:
            design_images = [*design_images, to_float(moved_view)]
            design_task += ("\nThe final image shows the requested camera viewpoint with unreconstructed regions. "
                            "Plan space for the visible subject and type around that view.")
        if design_input.get("has_failed_resize"):
            design_images.append(to_float(failed_resize))
        answer = proposal["design"] if proposal else (
            ask_stage(ai, "ad_layout", design_task, design_input, design_images) | frame_fields)
        try:
            layout = design_layers(answer, copy, size, settings["fonts"], result)
            effective = crop_subject(product, answer)
        except ValueError as error:
            if proposal:
                raise
            retry = design_input | {"previous_design": answer, "layout_error": str(error)}
            answer = ask_stage(ai, "ad_layout", design_task, retry, design_images) | frame_fields
            try:
                layout = design_layers(answer, copy, size, settings["fonts"], result)
            except ValueError:
                if format_rebuild is None:
                    raise
                # Start the placement check with known approved lettering when the proposed typography is invalid.
                approved = prior["design"]
                answer = answer | {"boxes": approved["boxes"], "styles": approved["styles"], "accent": None}
                layout = design_layers(answer, copy, size, settings["fonts"], result)
            effective = crop_subject(product, answer)

        composition_reason = proposal.get("composition_reason") if proposal else None
        offer_scene = None
        if product is not None and not proposal:
            blocking_images = design_images if format_rebuild is not None else images
            answer, layout, composition_reason = review_composition_plan(
                ai, product, answer, layout, result, design_input, blocking_images)
            offer_scene = review_offer_scene(ai, product, answer, layout, result, design_input, blocking_images)
            if offer_scene is not None:
                answer, layout, composition_reason = offer_scene
            effective = crop_subject(product, answer)
        detail_shot = product is not None and answer.get("source_crop") is not None
        alternate_framing = bool(scene_issue and detail_shot)
        if detail_shot and art_plan:
            art_plan = art_plan | {"shot_scale": "detail"}
            design_input["art_direction_plan"]["shot_scale"] = "detail"
        artwork = layout["artwork"]
        companion = None
        if not proposal:
            direction_input = {"message": brief["message"], "headline": copy["headline"], "idea": idea, "design": answer,
                               "product_facts": fields, "headline_locked": "headline" in fields or prior is not None}
            angle, scale = ANGLES[art_plan["angle"]], SHOT_SCALES[art_plan["shot_scale"]]
            lighting = art_plan["lighting"]
            direction_input.update(art_direction_plan=art_plan, angle_guide=angle, framing_guide=scale,
                                   lighting_guide=LIGHTING.get(lighting, lighting))
            if source_framing:
                direction_input["source_framing"] = source_framing
            if composition_reason:
                direction_input["approved_composition"] = composition_reason
            if revision_info:
                direction_input.update(revision_info)
            direction_task = PHOTO_DIRECTION + "\nKeep the selected angle, framing, and photo_device in the scene."
            if offer_scene is not None:
                direction_task += ("\nThe photograph continues under the name and price. Extend the same real scene "
                                   "through that area with readable, restrained texture and consistent light. Use the "
                                   "approved_composition rationale and keep the product at the placement shown.")
            if format_rebuild is not None:
                direction_input["target_photo_aspect"] = [artwork["w"], artwork["h"]]
                direction_input["approved_effect_hint"] = prior.get("revision", {}).get("visible_result")
                direction_input["approved_photo_cue"] = prior.get("photo_cue")
                direction_task += (" Arrange the subject and lighting for the new composition and art region. "
                                   "Use perspective, directional light or meaningful surroundings across the long axis. "
                                   "Follow the new placement preview and target_photo_aspect. Omit any description "
                                   "of aspect ratio from scene.")
            if detail_shot:
                direction_task += "\nThe approved composition uses a detail crop. Keep that magnification in the photograph."
            direction_images = images[:1]
            if (result / "composition.png").is_file():
                with Image.open(result / "composition.png") as proof:
                    direction_images.append(to_float(proof))
            elif (result / artwork["file"]).is_file():
                direction_images.append(to_float(compose(layout, result)))
            if len(direction_images) == 2:
                direction_task += ("\nThe second image is the approved placement before photography. Use the subject "
                                   "position and open regions from it. Its lettering and flat colors are placeholders.")
                direction_input["rendered_text_bounds"] = rendered_text_bounds(layout, result)
            if feedback:
                if alternate_framing:
                    direction_task += ("\nUse the requested photographic effect with the approved close framing. "
                                       "Omit any instruction to keep the entire product in view.")
                else:
                    direction_task += "\nDescribe the exact requested_scene_change and omit any conflicting design detail."
            if staged_view is not None:
                direction_images.append(staged_view)
                direction_task += ("\nThe rearrangement guide shows the actual product in its new positions. Keep each "
                                   "piece exactly once there and give all components realistic support on one surface.")
            elif moved_view is not None:
                direction_input["viewpoint"] = viewpoint
                direction_images.append(to_float(moved_view))
                direction_task += ("\nThe last image is the requested viewpoint, with gray gaps to reconstruct. "
                                   "Describe the new scene at that actual angle. Keep the visible product identity and "
                                   "use the planned quiet regions for type. The camera view takes priority over angle_guide.")
            if format_rebuild is not None:
                direction_images.append(prior_image)
                direction_task += (" The last image is the approved poster. Describe the specific use gesture or "
                                   "physical effect visible in its photograph that carries the message, and keep "
                                   "that effect on the same product in the new arrangement. The source photo and "
                                   "neutral placement preview identify the product and geometry; they omit the effect. "
                                   "Do not inherit the approved poster's framing, exact positions, typography or ratio. "
                                   "When approved_photo_cue is present, keep that physical relationship in scene.")
            if campaign_intent:
                direction_task += ("\nLay out a real scene for the hypothesis and the approved headline position. "
                                   "Describe the path, gesture or feature that points toward the planned design. "
                                   "Keep layout_device to guide photography only: include no graphic or text in the photo.")
            for direction_attempt in range(2):
                direction = ask_stage(ai, "ad_scene_direction", direction_task, direction_input, direction_images)
                if direction.get("companion") is None:
                    if direction_attempt:
                        raise ValueError("필수 소품의 수정된 촬영 위치를 확인하지 못했습니다")
                    break
                try:
                    lettering = rendered_text_bounds(layout, result)
                    companion = planned_companion(direction, artwork, None, size, lettering)
                    break
                except ValueError as error:
                    if direction_attempt:
                        raise
                    direction_input.update(previous_companion=direction.get("companion"), layout_error=str(error),
                                           rendered_text_bounds=lettering)
                    direction_task += "\nKeep the required companion and correct its box using the latest layout_error."
            answer = answer | {"art_direction": required_text(direction, "scene", "촬영 연출")}
            photo_cue = (direction_input.get("approved_photo_cue") or required_text(direction, "photo_cue", "촬영 연출"))
            if companion:
                photo_cue = (f"Show the complete {companion['object']} with clear space around its outline, "
                             f"separate from the product. Required visible detail: {companion['cue']}")

        picture_size = (artwork["w"], artwork["h"])
        subject_box = answer.get("subject_box")
        placement = ("Follow the reference's product placement and scale if it has a plain gray background. "
                     if subject_box else "")
        correction = (f"Required photographic change, taking precedence over other scene details: {feedback} "
                      "Keep all poster copy out of the photograph. " if feedback and not alternate_framing else "")
        guide = (f"{ANGLES[art_plan['angle']]} {SHOT_SCALES[art_plan['shot_scale']]} "
                 if art_plan else "")
        if art_plan and not alternate_framing and companion is None:
            guide += f"{art_plan['photo_device']} "
        prompt = ("Make a new commissioned photograph with a newly built setting and lighting. "
                  f"{correction}{placement}{guide}Replace the surroundings with the requested setting. "
                  "Keep the product shape, material, visible components and varieties shown in the reference. "
                  "Do not introduce additional products or interchangeable variants. "
                  f"{answer['art_direction']} Preserve the recognizable product details and genuine packaging labels. "
                  "Keep surroundings, props and surfaces free of stray lettering, numbers and logos. "
                  "Add no advertising text or overlays.")
        if answer["edge"] == "blend":
            prompt += f" Keep the outer background low-detail and close to the surrounding {answer['background']} color."
        if answer["crop"] == "preserve":
            prompt += " Keep all of the product in frame with space around its edges."
        if detail_shot:
            prompt += " Keep the same close framing of the product detail shown in the reference."
        if staged_view is not None:
            prompt = ("Make one finished commercial photograph. The first image defines product identity and details. "
                      "The second image defines the new product arrangement, scale and positions. Keep each piece once. "
                      "Do not restore moved pieces to their old positions. Replace gray and repair newly exposed "
                      f"surfaces with physically plausible material, support and shadows. {answer['art_direction']} "
                      "Preserve the actual product and packaging. Add no writing or graphics.")
        elif moved_view is not None:
            prompt = ("Make one finished commercial photograph. The first image shows the original item and its details. "
                      "Use the second image's camera viewpoint and visible arrangement. Gray gaps are missing pixels: "
                      "complete them naturally with details consistent with the first image. Keep the changed angle. "
                      f"{answer['art_direction']} Preserve the actual product, components and packaging. "
                      "Add no writing, graphics or products not specified in the scene.")
        if format_rebuild is not None:
            orientation = "vertical" if picture_size[0] < picture_size[1] else "horizontal"
            if picture_size[0] == picture_size[1]:
                orientation = "square"
            prompt = (f"Compose for this {orientation} photograph, filling its {picture_size[0]} by "
                      f"{picture_size[1]} frame with the planned composition. {prompt}")
        if photo_cue and companion is None:
            prompt += f" The finished photo must visibly show: {photo_cue}"
        if companion:
            left, top, width, height = companion["box"]
            prompt = (f"Photograph the product on its own first. Leave bare continuous supporting surface at "
                      f"{left}%–{left + width}% across and {top}%–{top + height}% down, with no {companion['object']} "
                      "or substitute there or elsewhere in frame. A secondary object will be added in a separate edit. "
                      f"{prompt}")
        if preview:
            if (result / artwork["file"]).is_file():
                with Image.open(result / artwork["file"]) as planned:
                    scene = planned.convert("RGB")
            else:
                scene = preview_artwork(photo, artwork, size, answer)
            final_review_reason = None
        else:
            model = models["generation"]
            model.use_lora(None, 0)
            generation = settings["generation"]
            render_size = picture_size
            if format_rebuild is not None:
                scale = min(1, sqrt(canvas["width"] * canvas["height"] / (picture_size[0] * picture_size[1])))
                render_size = tuple(round(side * scale) for side in picture_size)
            if staged_view is not None:
                placement_area = subject_box or [8, 8, 84, 84]
                guide = product_reference(photo, effective, render_size, placement_area)
                guide.save(result / "staging.png")
                if moved_piece is not None:
                    piece_box = frame_subject(moved_piece, picture_size, placement_area).getbbox()
                    revision_info["staged_piece_box"] = list(piece_box)
                    on_gray(moved_piece.crop(moved_piece.getbbox())).save(result / "staged-piece.png")
            elif moved_view is not None:
                placement_box = pixel_box(subject_box or [0, 0, 100, 100], picture_size)
                guide = place_camera_view(moved_view, source_mask.getbbox(), photo.size, picture_size, placement_box)
                guide.save(result / "viewpoint.png")
            if staged_view is not None or moved_view is not None:
                original = ImageOps.contain(photo, (640, 640), Image.Resampling.LANCZOS)

                def photograph(instruction):
                    return model.edit_with_reference(original, guide, instruction, generation["seed"], generation["steps"])
            else:
                reference = product_reference(photo, effective, render_size, subject_box, answer.get("source_crop"))

                def photograph(instruction):
                    return model.edit(reference, instruction, generation["seed"], generation["steps"], size=render_size)

            approved_detail = on_gray(effective) if detail_shot and effective is not None else None
            reshot_action = False
            reshot_background = False
            patch_bounds = None
            secondary_object = None

            def review_photo(candidate):
                picture = to_float(candidate)
                seen, message_observation = None, None
                message = {"message_supported": True}
                if photo_cue:
                    observed = ask_stage(ai, "ad_photo_observation", PHOTO_OBSERVATION, {}, [picture])
                    seen = required_text(observed, "observation", "사진 장면 관찰")
                    message = ask_stage(ai, "ad_photo_message", PHOTO_MESSAGE,
                                        {"observation": seen, "headline": copy["headline"],
                                         "message": brief.get("message", copy["headline"])}, [picture])
                    message_observation = required_text(message, "message_observation", "사진과 광고 문구 비교")
                # Check the observable motion from the reshoot without requiring the draft's exact grip or instant.
                photo_inputs = {"photo_cue": seen if reshot_action else photo_cue,
                                "has_approved_detail": approved_detail is not None,
                                "crop": answer["crop"], "product_facts": fields, "message": brief.get("message"),
                                "art_direction": answer["art_direction"]}
                if companion:
                    photo_inputs.update(required_companion={key: companion[key] for key in ("object", "cue")},
                                        owner_idea=idea)
                proof = [images[0], picture, *([to_float(approved_detail)] if approved_detail else []),
                         *photo_review_details(candidate)]
                check = ask_stage(ai, "ad_photo_cue_check", PHOTO_CUE_CHECK, photo_inputs, proof)
                required = ("stray_writing", "blocking_distraction") if reshot_background else ("stray_writing",)
                missing = " and ".join(key for key in required if key not in check)
                if missing:
                    check = ask_stage(ai, "ad_photo_cue_check", PHOTO_CUE_CHECK +
                                      f"\nThe previous answer omitted {missing}. Return it explicitly.", photo_inputs, proof)
                if reshot_background and "blocking_distraction" not in check:
                    raise ValueError("다시 촬영한 사진에서 배경 물체가 정리됐는지 확인하지 못했습니다")
                observation = required_text(check, "observation", "촬영 결과 검토")
                identity = check.get("identity_matches")
                identity_observation = required_text(check, "identity_observation", "촬영 결과 검토")
                visible = check.get("visible")
                supported = message.get("message_supported")
                if any(type(value) is not bool for value in (visible, identity, supported)):
                    raise ValueError("촬영한 사진에서 연출을 확인하지 못했습니다")
                mark = check.get("stray_writing")
                if "stray_writing" not in check or (mark is not None and not isinstance(mark, dict)):
                    raise ValueError("촬영한 사진에 새로 생긴 글자가 있는지 확인하지 못했습니다")
                written = required_text(mark, "observation", "사진 글자 확인") if mark is not None else None
                finding = {"visible": visible, "observation": observation, "identity_matches": identity,
                           "identity_observation": identity_observation, "message_supported": supported,
                           "observed_scene": seen, "message_observation": message_observation, "stray_writing": written}
                photo_checks.append(finding)
                return check, message, finding

            foreground = None
            base_scene = None
            companion_feedback = None
            for attempt in range(2):
                if companion:
                    if base_scene is None:
                        generated = photograph(prompt)
                        base_scene = ImageOps.fit(generated.convert("RGB"), picture_size, Image.Resampling.LANCZOS)
                        protected = find_foreground(base_scene, models["mask"], threshold)
                    patch_bounds, instruction, companion = plan_companion_insertion(
                        ai, base_scene, companion, protected, layout, result, companion_feedback,
                        rejected=scene if attempt else None)
                    original_scene = base_scene
                    secondary_object = companion["object"]
                    scene = repair_scene_patch(model, original_scene, patch_bounds, instruction, generation,
                                               secondary_object=secondary_object, foreground=protected,
                                               companion_mask=lambda patch: product_mask(
                                                   models["mask"], to_float(patch)) > threshold)
                elif patch_bounds is None:
                    generated = photograph(prompt)
                    scene = ImageOps.fit(generated.convert("RGB"), picture_size, Image.Resampling.LANCZOS)
                else:
                    original_scene = scene
                    scene = repair_scene_patch(model, original_scene, patch_bounds, prompt, generation,
                                               secondary_object=secondary_object, foreground=protected)
                check, message, finding = review_photo(scene)
                if (not attempt and kept is None and "headline" not in fields and finding["visible"]
                        and finding["identity_matches"] and not finding["message_supported"]
                        and not finding["stray_writing"] and not blocking_distraction(check)
                        and message.get("repair_kind") == "scene"):
                    reframed = reconcile_draft_headline(ai, fields, idea, brief, images[0], to_float(scene), finding)
                    if reframed is not None:
                        revised_brief, observation, line_break = reframed
                        revised_copy = copy | {"headline": revised_brief["headline"]}
                        style = answer["styles"]["headline"].copy()
                        for key in ("lines", "break_after", "emphasis"):
                            style.pop(key, None)
                        if line_break is not None and style["max_lines"] == 2:
                            style["break_after"] = line_break
                        revised_answer = answer | {"styles": answer["styles"] | {"headline": style}}
                        with TemporaryDirectory(dir=result) as draft_folder:
                            staged = Path(draft_folder)
                            try:
                                revised_texts = layout_texts(revised_answer, revised_copy, size, settings["fonts"])
                                title = next(layer for layer in revised_texts if layer["role"] == "headline")
                                render_texts([title], size, staged)
                            except ValueError:
                                pass
                            else:
                                for layer in staged.iterdir():
                                    layer.replace(result / layer.name)
                                finding["draft_copy_revision"] = {"previous_headline": copy["headline"],
                                                                  "mismatch": finding["message_observation"]}
                                finding.update(message_supported=True, message_observation=observation)
                                brief, copy, answer = revised_brief, revised_copy, revised_answer
                                layout["texts"] = [title if layer["role"] == "headline" else layer
                                                   for layer in layout["texts"]]
                                design_input.update(brief=brief, copy=copy, headline_breaks=headline_breaks(copy["headline"]))
                matches = finding["visible"] and finding["identity_matches"] and finding["message_supported"]
                if matches and finding["stray_writing"]:
                    mark = check["stray_writing"]
                    foreground = find_foreground(scene, models["mask"], threshold)
                    fixed = remove_photo_writing(scene, foreground, mark)
                    if fixed is not None:
                        scene = fixed
                        check, message, finding = review_photo(scene)
                        matches = finding["visible"] and finding["identity_matches"] and finding["message_supported"]
                if matches and not finding["stray_writing"]:
                    if patch_bounds is not None:
                        repair = ask_stage(ai, "ad_photo_patch_check", PHOTO_PATCH_CHECK,
                                           {"photo_cue": photo_cue, "secondary_object": secondary_object,
                                            **({"measure_object": True} if companion else {})},
                                           [to_float(original_scene), to_float(scene), to_float(scene.crop(patch_bounds))])
                        finding["scene_patch_review"] = required_text(repair, "observation", "사진 부분 수정 검토")
                        patch_accepted = repair.get("accepted") is True
                        if companion and patch_accepted:
                            patch_accepted = companion_has_visible_scale(repair.get("object_box"))
                            if not patch_accepted:
                                finding["scene_patch_review"] = ("필수 소품이 편집 영역에 비해 너무 작거나 "
                                                                 "전체 윤곽을 확인할 수 없습니다")
                        if not patch_accepted:
                            if companion and not attempt:
                                companion_feedback = finding["scene_patch_review"]
                                continue
                            raise ValueError(f"사진 부분 수정이 자연스럽지 않습니다. {finding['scene_patch_review']}")
                    obstruction = blocking_distraction(check)
                    if reshot_background and check.get("blocking_distraction") is not None and obstruction is None:
                        raise ValueError("다시 촬영한 사진에서 배경 물체가 정리됐는지 확인하지 못했습니다")
                    if obstruction is not None:
                        observation, corrected_scene = obstruction
                        finding["blocking_distraction"] = observation
                        if companion:
                            raise ValueError(f"촬영한 사진에 상품과 경쟁하는 배경 물체가 남았습니다. {observation}")
                        if attempt:
                            raise ValueError(f"촬영한 사진에 상품과 경쟁하는 배경 물체가 남았습니다. {observation}")
                        answer = answer | {"art_direction": corrected_scene}
                        photographic_goal = f" The photograph must show: {photo_cue}" if photo_cue else ""
                        prompt = ("Priority reshoot correction. Make one finished commercial photograph from the "
                                  "provided product reference. Keep its identity, product arrangement, scale and "
                                  f"camera framing. {corrected_scene} Preserve quiet space for later typography. "
                                  f"Use only the named scene elements. Add no advertising text or graphics.{photographic_goal}")
                        foreground = None
                        reshot_background = True
                        continue
                    if reshot_action:
                        photo_cue = finding["observed_scene"]
                    break
                foreground = None
                if companion and not attempt and finding["identity_matches"]:
                    companion_feedback = finding["stray_writing"] or (
                        finding["observation"] if finding["message_supported"] else finding["message_observation"])
                    continue
                if companion and not finding["identity_matches"]:
                    raise ValueError(f"촬영한 사진의 상품이 원본과 다릅니다. {finding['identity_observation']}")
                if attempt:
                    if not finding["identity_matches"]:
                        raise ValueError(f"촬영한 사진의 상품이 원본과 다릅니다. {finding['identity_observation']}")
                    if not finding["message_supported"]:
                        raise ValueError(f"사진이 광고 문구를 뒷받침하지 않습니다. {finding['message_observation']}")
                    if finding["stray_writing"]:
                        raise ValueError(f"촬영한 사진에 의도하지 않은 글자가 남았습니다. {finding['stray_writing']}")
                    raise ValueError(f"사진에 의도한 연출이 보이지 않습니다. {finding['observation']}")
                wrong_message = not finding["message_supported"]
                instructions = message if wrong_message else check
                correction = required_text(instructions, "correction", "촬영 결과 검토")
                reposition = instructions.get("reposition")
                if type(reposition) is not bool:
                    raise ValueError("촬영을 고칠 때 상품 배치를 바꿔야 하는지 확인하지 못했습니다")
                if wrong_message:
                    if not finding["identity_matches"]:
                        correction += " " + required_text(check, "correction", "상품 원형 보존")
                    photo_cue = required_text(message, "corrected_photo_cue", "광고 문구에 맞춘 촬영 수정")
                    repair_kind = message.get("repair_kind", "action")
                    if repair_kind == "action":
                        support = required_text(message, "visible_support", "움직이는 부분을 잡는 곳")
                        target = required_text(message, "action_target", "동작이 닿는 곳")
                        photo_cue = f"{photo_cue} Visible support: {support} Visible destination: {target}"
                        reshot_action = True
                    elif repair_kind != "scene":
                        raise ValueError("광고 문구에 맞춘 장면 수정 방식을 확인하지 못했습니다")
                    answer = answer | {"art_direction": correction}
                    box = message.get("repair_box")
                    if repair_kind == "scene" and not reposition and finding["identity_matches"] and box is not None:
                        validate_boxes({"repair": box}, ("repair",))
                        x, y, width, height = pixel_box(box, scene.size)
                        protected = find_foreground(scene, models["mask"], threshold)
                        if min(width, height) < 2 or protected[y:y + height, x:x + width].any():
                            raise ValueError("사진 부분 수정 영역이 상품과 겹치거나 너무 작습니다")
                        if message.get("secondary_object") is not None:
                            secondary_object = required_text(message, "secondary_object", "사진의 보조 소품")
                        patch_bounds = (x, y, x + width, y + height)
                        instruction = message.get("patch_instruction")
                        prompt = f"{correction} The repaired photograph must show: {photo_cue}"
                        if secondary_object and isinstance(instruction, str) and instruction.strip():
                            prompt = instruction.strip()
                        continue
                if wrong_message or (reposition and staged_view is None and moved_view is None):
                    if reposition and staged_view is None and moved_view is None:
                        reference = photo.convert("RGB")
                    camera = (f"{ANGLES[art_plan['angle']]} {SHOT_SCALES[art_plan['shot_scale']]} "
                              f"{LIGHTING.get(art_plan['lighting'], art_plan['lighting'])} " if art_plan else "")
                    photographic_goal = f" The photograph must show: {photo_cue}" if photo_cue else ""
                    prompt = ("Make one finished commercial photograph. The reference identifies the original item, "
                              "its proportions, materials, labels, and visible parts. Change the arrangement to the "
                              f"one described here: {correction}{photographic_goal} "
                              f"{camera} Keep a quiet area for the later lettering. Use a background that can sit "
                              f"against {answer['background']}. Preserve product identity. Add no text or graphics.")
                else:
                    prompt = f"Priority reshoot correction, replacing any conflicting placement instruction: {correction} {prompt}"
            if foreground is None:
                foreground = find_foreground(scene, models["mask"], threshold)
            distractions = check.get("distractions")
            if isinstance(distractions, list) and 1 <= len(distractions) <= 2:
                layout["photo_regions"] = find_photo_regions(ai, scene, photo_cue)
                scene, cleanup = clean_photo_distractions(ai, scene, foreground, layout["photo_regions"],
                                                          distractions, brief.get("message"))
                if cleanup:
                    photo_checks[-1]["cleanup"] = cleanup
            if answer["edge"] == "blend":
                scene = blend_edges(scene, foreground, position=(artwork["x"], artwork["y"]), canvas_size=size)
            scene.save(result / artwork["file"])
            Image.fromarray(foreground).save(result / "foreground.png")
            if photo_cue:
                layout["photo_cue"] = photo_cue
            answer, layout, final_review_reason = review_finished(ai, answer, layout, scene, foreground, result,
                                                                  design_input, images[1:],
                                                                  assess_composition=bool(art_plan and not proposal))
            answer, layout, offer_reason = review_crowded_offer(
                ai, answer, layout, foreground, result, settings["fonts"])
            final_review_reason = offer_reason or final_review_reason
        layout.update(fields=fields, brief=brief, design=answer, scene_prompt=prompt, preview=preview,
                      final_review_reason=final_review_reason)
        if art_plan:
            layout["art_direction_plan"] = art_plan
        if photo_cue:
            layout["photo_cue"] = photo_cue
        if companion:
            layout["photo_companion"] = companion
        if photo_checks:
            layout["photo_cue_checks"] = photo_checks
        if source_framing:
            layout["source_framing"] = source_framing
        if composition_reason:
            layout["composition_reason"] = composition_reason
        if preview:
            scene.save(result / "scene.png")
        poster = compose(layout, result)
        poster.save(result / "ad.png")
        if revision_info:
            layout["revision"] = revision_info
        if viewpoint:
            layout["viewpoint"] = viewpoint
        save_layout(layout, result)
        result.rename(folder)
    return layout


def main():
    from .make_ad import load_models
    from ..photo_retouch.interpret import Interpreter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", type=Path, help="파일명, 장면 한 줄, 상품 정보를 담은 JSON")
    parser.add_argument("photos", type=Path, help="참고 사진 폴더")
    parser.add_argument("output", type=Path, help="상품별 장면과 글자 레이어를 저장할 폴더")
    parser.add_argument("--references", nargs="+", type=Path, default=[], metavar="PHOTO", help="포스터 구도 참고 이미지")
    parser.add_argument("--preview", action="store_true", help="원본 사진으로 구도 시안 만들기")
    drafts = parser.add_mutually_exclusive_group()
    drafts.add_argument("--designs", type=Path, help="상품별로 저장된 구도 시안을 그대로 사용하기")
    drafts.add_argument("--revise", type=Path, help="이 폴더의 완성본을 보고 사진부터 다시 생성해 비교하기")
    parser.add_argument("--rejected", type=Path, help="다음 개정에서 피해야 할 보류 후보 폴더")
    parser.add_argument("--low-vram", action="store_true", help="가중치를 나눠 올려 그래픽 메모리를 절약하기")
    args = parser.parse_args()
    if args.rejected and not args.revise:
        parser.error("--rejected에는 --revise가 필요합니다")
    items = json.loads(args.items.read_text(encoding="utf-8"))
    references = []
    for path in args.references:
        with Image.open(path) as source:
            references.append(ImageOps.exif_transpose(source).convert("RGB"))
    models = {"ai": Interpreter()} if args.preview else load_models(low_vram=args.low_vram)
    for index, item in enumerate(items, 1):
        reference = Path(item["file"])
        folder = args.output / f"{index:02d}_{reference.stem}"
        with Image.open(args.photos / reference) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        proposal = load_layout(args.designs / folder.name) if args.designs else None
        previous = args.revise / folder.name if args.revise else None
        rejected = args.rejected / folder.name if args.rejected else None
        if rejected and not rejected.is_dir():
            rejected = None
        make_scene_ad(photo, item["fields"], item.get("idea", ""), models, folder,
                      references=references, preview=args.preview, proposal=proposal, previous=previous,
                      rejected=rejected)
        print(folder / "ad.png", flush=True)


if __name__ == "__main__":
    main()
