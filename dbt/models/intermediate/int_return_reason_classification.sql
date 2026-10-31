select
    return_id,
    order_id,
    return_reason,
    {{ get_return_reason_class('return_reason') }} as return_reason_class,
    returned_at
from {{ ref('stg_returns') }}
