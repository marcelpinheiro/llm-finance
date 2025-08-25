with source as (
    select * from {{ source('finance', 'companies') }}
),

renamed as (
    select
        id,
        symbol,
        name,
        sector,
        industry,
        market_cap,
        created_at
    from source
)

select * from renamed
