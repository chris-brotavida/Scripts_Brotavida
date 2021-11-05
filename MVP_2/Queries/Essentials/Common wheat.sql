-- Common wheat (NO FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content
FROM foodb.contents cont
WHERE cont.food_id = 187 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.orig_food_common_name = 'Wheat'
AND cont.orig_food_part LIKE '%seed%' -- '%plant%'
-- GROUP BY cont.orig_food_part
ORDER BY cont.source_id;