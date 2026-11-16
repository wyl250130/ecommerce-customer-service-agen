# 聊天验收事实库

`customer_service_train.sqlite` 是项目原有的样本级模拟事实库。
`customer_service_chat_e2e.sqlite` 由 `scripts/build_chat_e2e_facts.py` 生成，
用于连续询问同一订单的发货、预计送达和商品信息。

生成脚本从原库选择一条同快照内已有订单、商品、SKU 和物流记录的样本，
复制原库后派生 `CHAT-E2E-SNAPSHOT-001`。只有预计送达文本是额外模拟值，
可在 `chat_fixture_metadata.simulated_fields_json` 中核对。原库不会被修改。

本库仍不包含可靠的真实用户 ID、下单数量或成交单价，不能用来证明真实顾客
拥有该订单，也不能把预计送达文本当作真实物流承诺。
