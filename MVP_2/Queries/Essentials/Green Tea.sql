-- Green Tea (NO FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content) AS orig_content, 
	AVG(cont.orig_min) AS orig_min, AVG(cont.orig_max) AS orig_max, cont.orig_unit, 
	AVG(cont.standard_content) AS standard_content
FROM foodb.contents cont
WHERE cont.food_id = 940 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.orig_food_common_name LIKE '%leaves%'
GROUP BY cont.source_id 
ORDER BY cont.source_id;