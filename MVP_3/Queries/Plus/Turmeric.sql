-- Turmeric (NO FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content
FROM foodb.contents cont
WHERE cont.food_id = 68 AND cont.source_type = 'Compound' 
AND cont.source_id IN (453, 455, 571, 572, 1977, 2251, 3011, 3192, 7058, 
					   9021, 11798, 11817, 12295, 12741, 13231, 13570, 
					   14800, 16599, 23102, 23288, 24066, 30890)
AND cont.preparation_type LIKE '%dried%'
ORDER BY cont.source_id;