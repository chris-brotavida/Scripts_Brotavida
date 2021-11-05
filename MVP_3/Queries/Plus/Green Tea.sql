-- Green Tea (NO FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content) AS orig_content, 
	AVG(cont.orig_min) AS orig_min, AVG(cont.orig_max) AS orig_max, cont.orig_unit, 
	AVG(cont.standard_content) AS standard_content
FROM foodb.contents cont
WHERE cont.food_id = 940 AND cont.source_type = 'Compound' 
AND cont.source_id IN (453, 455, 571, 572, 1977, 2251, 3011, 3192, 7058, 
					   9021, 11798, 11817, 12295, 12741, 13231, 13570, 
					   14800, 16599, 23102, 23288, 24066, 30890)
AND cont.orig_food_common_name LIKE '%leaves%'
GROUP BY cont.source_id 
ORDER BY cont.source_id;