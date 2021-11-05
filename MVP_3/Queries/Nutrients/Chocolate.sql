-- Chocolate (FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content) AS orig_content, 
	AVG(cont.orig_min) AS orig_min, AVG(cont.orig_max) AS orig_max, cont.orig_unit, 
	AVG(cont.standard_content) AS standard_content
FROM foodb.contents cont
WHERE cont.food_id = 709 AND cont.source_type = 'Nutrient' 
AND cont.source_id IN (2, 3)
AND cont.orig_food_common_name LIKE '%milk'
GROUP BY source_id 
ORDER BY source_id;