with stg_companies as (
    select * from {{ ref('stg_companies') }}
),

final as (
    select
        id as company_id,
        symbol,
        name,
        sector,
        industry,
        market_cap,
        created_at
    from stg_companies
)

select * from final
