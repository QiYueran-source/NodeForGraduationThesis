-- 原子化：读 key 的 JSON 数组，若 node_id 不在列表中则追加并写回
-- KEYS[1]: counter key
-- ARGV[1]: node_id（当前节点名）
-- 返回: 1=已追加, 0=已在列表中无需追加
-- 依赖 Redis 7+ 的 cjson
local raw = redis.call('GET', KEYS[1])
local list = {}
if raw and raw ~= '' then
  list = cjson.decode(raw)
end
if type(list) ~= 'table' then
  list = {}
end
for i, v in ipairs(list) do
  if v == ARGV[1] then
    return 0
  end
end
table.insert(list, ARGV[1])
redis.call('SET', KEYS[1], cjson.encode(list))
return 1
