# python agentic_ai_demo.py 
Available tools:
- search_knowledge_base: Look up policy or buying guidance.
- search_products: Find products that match category and budget constraints.
- calculate_shipping: Estimate shipping cost from weight and speed.

========================================================================

Agentic AI Demo

========================================================================

## Task: Recommend a laptop bag under $80 and estimate expedited shipping for 2 kg.

### [Step 1] Thought
  I should gather domain guidance before choosing a product.
[Step 1] Action
  search_knowledge_base({'query': 'Recommend a laptop bag under $80 and estimate expedited shipping '
          'for 2 kg.'})
### [Step 1] Observation
  {'matches': [{'summary': 'Standard shipping takes 3-5 business days. Expedited '
                         'shipping takes 1-2 business days.',
              'topic': 'shipping'},
             {'summary': 'Laptop bags are best filtered by budget, laptop '
                         'size, and travel or commute needs.',
              'topic': 'bags'}]}

### [Step 2] Thought
  I now have constraints, so I should find matching products.
### [Step 2] Action
  search_products({'category': 'bag', 'max_price': 80.0})
### [Step 2] Observation
  {'count': 2,
 'products': [{'category': 'bag',
               'features': ['lightweight', 'laptop', 'commute'],
               'name': 'CityLite 14',
               'price': 49.0,
               'weight_kg': 0.9},
              {'category': 'bag',
               'features': ['laptop', 'travel', 'water-resistant'],
               'name': 'TravelPro 15',
               'price': 79.0,
               'weight_kg': 1.2}]}

### [Step 3] Thought
  The user asked about shipping, so I should estimate delivery cost.
[Step 3] Action
  calculate_shipping({'expedited': True, 'weight_kg': 2.0})
[Step 3] Observation
  {'estimated_delivery': '1-2 business days',
 'expedited': True,
 'shipping_cost_usd': 15.99}

## Final answer
------------------------------------------------------------------------
- Recommended product: CityLite 14 at $49.00.
- Why: it fits the inferred category 'bag' and is the lowest-priced match.
- Features: lightweight, laptop, commute.
- Policy guidance: Standard shipping takes 3-5 business days. Expedited shipping takes 1-2 business days.
- Shipping estimate: $15.99, 1-2 business days.
- Original request: Recommend a laptop bag under $80 and estimate expedited shipping for 2 kg.

