import torch

def generate_recipe_summary(
    recipe_texts: list[str],
    model,
    processor,
    max_new_tokens: int = 512
) -> str:
    recipes_combined = ""
    for i, recipe in enumerate(recipe_texts[:3], 1):
        recipes_combined += f"\n\n--- RECIPE {i} ---\n{recipe}"

    prompt = f"""You are a helpful culinary assistant. Below are {len(recipe_texts[:3])} recipes. 
Please provide a brief markdown summary with:
- A short 1-2 sentence overview of each recipe
- Key ingredients highlighted
- Estimated difficulty (Easy/Medium/Hard)
- Which recipe might be best for a quick weeknight dinner

For example use the following format: 

```markdown
# Recipe summary

## <recipe_name>

[details]
```

Keep the summary concise and well-formatted in markdown. Return in ```markdown``` tags so it can be easily parsed.

<recipes>
{recipes_combined}
</recipes>

## Summary:"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return output_text.strip()

def _markdown_to_simple_html(markdown_text: str, max_reviews: int = 1) -> str:
    lines = markdown_text.strip().split('\n')
    title = ""
    description = ""
    recipe_id = ""
    cook_time = ""
    num_ratings = ""
    ingredients = []
    steps = []
    tags = []
    reviews = []

    current_section = None
    in_ingredients = False
    in_steps = False
    in_reviews = False
    in_tags = False
    review_count = 0

    for line in lines:
        line = line.strip()
        if line.startswith('# ') and not title:
            title = line[2:].strip()
            continue
        if line.startswith('**ID:**'):
            recipe_id = line.replace('**ID:**', '').strip()
            continue
        if line.startswith('**Time:**'):
            cook_time = line.replace('**Time:**', '').strip()
            continue
        if line.startswith('**Number of Ratings:**'):
            num_ratings = line.replace('**Number of Ratings:**', '').strip()
            continue
        if line.startswith('## '):
            section_name = line[3:].strip().lower()
            in_ingredients = section_name == 'ingredients'
            in_steps = section_name.startswith('steps')
            in_reviews = section_name == 'reviews'
            in_tags = section_name == 'tags'
            current_section = section_name
            continue
        if current_section == 'description' and line and not line.startswith('#'):
            description = line
            continue
        if in_ingredients and line.startswith('- '):
            ingredients.append(line[2:].strip())
            continue
        if in_steps and line and line[0].isdigit():
            step_text = line.split('. ', 1)[-1] if '. ' in line else line
            steps.append(step_text.strip())
            continue
        if in_tags and line.startswith('`'):
            tag_list = [t.strip().strip('`') for t in line.split(',')]
            tags.extend(tag_list)
            continue
        if in_reviews and line.startswith('> ') and review_count < max_reviews:
            reviews.append(line[2:].strip())
            review_count += 1
            continue

    html = f'''
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 4px; background: #fff; font-family: system-ui, -apple-system, sans-serif; font-size: 12px; height: 400px; overflow-y: auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-weight: bold; font-size: 14px; color: #333; margin-bottom: 8px;">{title}</div>
        <div style="display: flex; gap: 12px; font-size: 11px; color: #666; margin-bottom: 10px; flex-wrap: wrap;">
            {f'<span>⏱️ {cook_time}</span>' if cook_time else ''}
            {f'<span>⭐ {num_ratings} ratings</span>' if num_ratings else ''}
            {f'<span style="color: #999;">ID: {recipe_id}</span>' if recipe_id else ''}
        </div>
        <div style="color: #555; margin-bottom: 12px; font-style: italic; line-height: 1.4;">{description[:150]}{"..." if len(description) > 150 else ""}</div>
        <div style="margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 11px; color: #333; margin-bottom: 4px;">📝 Ingredients</div>
            <div style="color: #444; line-height: 1.5;">{", ".join(ingredients[:8])}{"..." if len(ingredients) > 8 else ""}</div>
        </div>
        <div style="margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 11px; color: #333; margin-bottom: 4px;">👨‍🍳 Steps ({len(steps)} total)</div>
            <ol style="margin: 0; padding-left: 20px; color: #444; line-height: 1.5;">
                {"".join(f'<li style="margin-bottom: 4px;">{step[:80]}{"..." if len(step) > 80 else ""}</li>' for step in steps[:4])}
                {f'<li style="color: #999;">...and {len(steps) - 4} more steps</li>' if len(steps) > 4 else ''}
            </ol>
        </div>
    '''
    if tags:
        display_tags = tags[:5]
        html += f'''
        <div style="margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 11px; color: #333; margin-bottom: 4px;">🏷️ Tags</div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {"".join(f'<span style="background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 10px;">{tag}</span>' for tag in display_tags)}
                {f'<span style="color: #999; font-size: 10px;">+{len(tags) - 5} more</span>' if len(tags) > 5 else ''}
            </div>
        </div>
        '''
    if reviews:
        html += f'''
        <div style="border-top: 1px solid #eee; padding-top: 10px; margin-top: 10px;">
            <div style="font-weight: bold; font-size: 11px; color: #333; margin-bottom: 4px;">💬 Review</div>
            <div style="color: #555; font-size: 11px; line-height: 1.4; background: #f9f9f9; padding: 8px; border-radius: 4px; font-style: italic;">"{reviews[0][:200]}{"..." if len(reviews[0]) > 200 else ""}"</div>
        </div>
        '''
    html += '</div>'
    return html

def create_recipe_cards_html(scores_and_samples: list[dict], num_results: int = 3) -> str:
    recipe_cards_html = []
    for item in scores_and_samples[:num_results]:
        sample = item["sample"]
        markdown_text = sample.get("recipe_markdown", "")
        card_html = _markdown_to_simple_html(markdown_text)
        recipe_cards_html.append(f'<div style="flex: 1; min-width: 0;">{card_html}</div>')
    combined_html = f'''
    <div style="margin-top: 16px;">
        <h3 style="font-family: system-ui, -apple-system, sans-serif; font-size: 16px; font-weight: 600; color: #333; margin-bottom: 12px;">Retrieved Texts</h3>
        <div style="display: flex; flex-direction: row; gap: 12px; width: 100%;">
            {"".join(recipe_cards_html)}
        </div>
    </div>
    '''
    return combined_html
