-- Common wheat (FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content
FROM foodb.contents cont
WHERE cont.food_id = 187 AND cont.source_type = 'Nutrient' 
AND cont.source_id IN (2, 3)
AND cont.orig_food_common_name = 'Wheat'
AND cont.orig_food_part LIKE '%seed%' 
ORDER BY cont.source_id;